import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensions of embeddings
embedding_dim = 768

df = pd.read_parquet("data/final_data_cleaned.parquet", engine='pyarrow')

def split_dataframe(df, random_seed=42):
    # For reproducibility
    np.random.seed(random_seed)

    df_train = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)
    
    disease_counts = df.groupby('disease').size()
    
    processed_diseases = set()
    
    # For each possible number of prescriptions (1 to 10)
    for n_prescriptions in range(1, 11):
        # Get diseases with exactly n prescriptions
        diseases_with_n = disease_counts[disease_counts == n_prescriptions].index.tolist()
        
        if len(diseases_with_n) > 0:
            # Calculate number of diseases to move to test set (10%, minimum 1)
            n_to_test = max(1, int(np.ceil(len(diseases_with_n) * 0.1)))
            
            # Randomly select diseases for test set
            test_diseases = np.random.choice(diseases_with_n, 
                                          size=n_to_test, 
                                          replace=False)
            
            # Add to test set
            test_mask = df['disease'].isin(test_diseases)
            df_test = pd.concat([df_test, df[test_mask]])
            
            # Add remaining to train set
            train_mask = df['disease'].isin(diseases_with_n) & ~df['disease'].isin(test_diseases)
            df_train = pd.concat([df_train, df[train_mask]])
            
            # Add to processed diseases
            processed_diseases.update(diseases_with_n)
    
    assert len(processed_diseases) == len(disease_counts), "Not all diseases were processed"
    assert len(df) == len(df_train) + len(df_test), "Row counts don't match"
    
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

df, df_test_filtered = split_dataframe(df, random_seed=42)

# Shuffle df_train and df_test
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_test_filtered = df_test_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

print("Head of the test set:")
print(df_test_filtered.head())

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh()
        )
    
    def forward(self, disease_embedding):
        return self.model(disease_embedding)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disease_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.drug_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, disease_embedding, drug_embedding):
        disease_features = self.disease_encoder(disease_embedding)
        drug_features = self.drug_encoder(drug_embedding)
        combined = torch.cat([disease_features, drug_features], dim=1)
        return self.classifier(combined)
    
class DrugGAN:
    def __init__(self, df, df_test_filtered, batch_size=32, lr_g=0.0001, lr_d=0.0001):
        self.df = df
        self.df_test_filtered = df_test_filtered
        self.batch_size = batch_size
        self.device = device
        
        # Convert embeddings to tensors and move to device
        self.disease_embeddings = torch.tensor(
            np.vstack(df["disease_embedding"].values), 
            dtype=torch.float32
        ).to(device)
        self.drug_embeddings = torch.tensor(
            np.vstack(df["drug_embedding"].values), 
            dtype=torch.float32
        ).to(device)
        
        # Normalize embeddings
        self.disease_embeddings = nn.functional.normalize(self.disease_embeddings, dim=1)
        self.drug_embeddings = nn.functional.normalize(self.drug_embeddings, dim=1)

        # Convert and normalize test embeddings
        self.disease_embeddings_test = torch.tensor(
            np.vstack(df_test_filtered["disease_embedding"].values), 
            dtype=torch.float32
        ).to(device)
        self.drug_embeddings_test = torch.tensor(
            np.vstack(df_test_filtered["drug_embedding"].values), 
            dtype=torch.float32
        ).to(device)
        self.disease_embeddings_test = nn.functional.normalize(self.disease_embeddings_test, dim=1)
        self.drug_embeddings_test = nn.functional.normalize(self.drug_embeddings_test, dim=1)
        
        # Create mappings
        self.disease_to_drugs = self._create_disease_drug_mapping(self.df)
        self.disease_to_drugs_test = self._create_disease_drug_mapping(self.df_test_filtered)

        self.disease_drug_map_test = {}

        for disease in self.df_test_filtered['disease'].unique():
            disease_drugs = self.df_test_filtered[self.df_test_filtered['disease'] == disease]['drug'].unique()
            self.disease_drug_map_test[disease] = set(disease_drugs)
        
        # Initialize models
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=lr_g,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=lr_d,
            betas=(0.5, 0.999)
        )
        
        # Use different loss functions
        self.adversarial_loss = nn.BCELoss()

        # Add variables for tracking best similarity
        self.best_similarity = -float('inf')
        self.best_generator_state = None
        self.stagnant_epochs = 0
    
    def _create_disease_drug_mapping(self, df):
        """Create a mapping of diseases to their valid drug embeddings"""
        disease_drug_map = {}
        for disease in df['disease'].unique():
            disease_drugs = df[df['disease'] == disease]['drug_embedding'].values
            embeddings = np.vstack(disease_drugs)
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            disease_drug_map[disease] = torch.tensor(
                embeddings,
                dtype=torch.float32
            ).to(self.device)
        return disease_drug_map
    
    def triplet_loss(self, anchor, positive, negatives, margin=0.3):
        """
        Compute triplet loss with cosine similarity for top-k hard negatives.
        
        This function encourages the model to bring the anchor (generated drug embedding)
        closer to the positive (valid drug for the disease) and farther from the 
        negatives (invalid or less relevant drugs), with a specified margin.
        
        - anchor: embedding of the generated drug
        - positive: embedding of a valid drug for the target disease
        - negatives: embeddings of top-k hard negative drugs
        - margin: desired minimum gap between positive and negative distances
        
        The loss is computed as:
            loss = mean(max(0, (1 - sim(anchor, positive)) - (1 - sim(anchor, negative)) + margin))
                = mean(max(0, sim(anchor, negative) - sim(anchor, positive) + margin))
        
        This encourages the similarity to the positive to be greater than the similarity 
        to any negative by at least the given margin.
        """
        # Normalize embeddings
        anchor = nn.functional.normalize(anchor, dim=1)
        positive = nn.functional.normalize(positive, dim=1)
        negatives = nn.functional.normalize(negatives, dim=1)
        
        # Compute similarities
        positive_sim = torch.cosine_similarity(anchor, positive)
        negative_sims = torch.cosine_similarity(anchor, negatives)  # Multiple negatives
        
        # Convert similarities to distances (1 - similarity)
        positive_dist = 1 - positive_sim
        negative_dists = 1 - negative_sims  # Distance for all negatives
        
        # Compute loss for each negative
        losses = torch.clamp(positive_dist - negative_dists + margin, min=0)
        
        # Average over all top-k negatives
        return losses.mean()
    
    def compute_similarity_reward(self, generated_embedding, disease_name, test=False):
        """
        Compute contrastive similarity reward using cosine similarity.
        
        This function evaluates how well the generated drug embedding aligns with 
        valid drugs for a given disease (positives) while penalizing similarity 
        to unrelated drugs (negatives).
        
        - generated_embedding: the embedding of the generated drug
        - disease_name: disease used to determine positive/negative sets
        - test: flag to indicate if test or training data should be used
        
        The reward is computed as:
            reward = mean(similarity with valid drugs) - 0.75 * mean(similarity with invalid drugs)
        
        A higher reward indicates that the model generates embeddings closer to the 
        correct drug space while being dissimilar to irrelevant ones.
        """

        valid_embeddings = (self.disease_to_drugs_test if test else self.disease_to_drugs)[disease_name]
        
        # Get positive samples (valid drugs for this disease)
        positive_similarities = torch.cosine_similarity(
            generated_embedding.unsqueeze(0),
            valid_embeddings
        )
        
        # Get negative samples (random drugs not valid for this disease)
        all_drugs = self.drug_embeddings
        # Create indices tensor
        disease_indices = torch.tensor([
            i for i, d in enumerate(self.df['disease']) if d == disease_name
        ], dtype=torch.long, device=self.device)
        
        # Create negative mask
        negative_mask = torch.ones(len(all_drugs), dtype=torch.bool, device=self.device)
        negative_mask[disease_indices] = False
        negative_embeddings = all_drugs[negative_mask]
        
        # Handle case where there might be no negative samples
        if len(negative_embeddings) == 0:
            return positive_similarities.mean(), positive_similarities.mean()
        
        negative_similarities = torch.cosine_similarity(
            generated_embedding.unsqueeze(0),
            negative_embeddings
        )
        
        # Compute contrastive reward
        positive_reward = positive_similarities.mean()
        negative_penalty = negative_similarities.mean()
        
        # Reward higher similarity to valid drugs and lower similarity to invalid ones
        reward = positive_reward - 0.75 * negative_penalty
        
        return reward, positive_reward
    
    def find_most_similar_drugs(self, disease_embedding, combined_df, device, top_n=5):
        # Generate the disease embedding
        disease_embedding = torch.tensor(disease_embedding, dtype=torch.float32).to(device)
        disease_embedding = disease_embedding.unsqueeze(0)
        
        # Generate drug embedding
        generated_drug_embedding = self.generate_drug(disease_embedding).cpu().numpy()
        
        # Normalize all drug embeddings
        drug_embeddings = np.vstack(combined_df["drug_embedding"].values)
        drug_embeddings = normalize(torch.tensor(drug_embeddings), dim=1).numpy()
        
        # Calculate cosine similarity
        similarities = cosine_similarity(generated_drug_embedding, drug_embeddings)
        
        # Get indices sorted by similarity
        sorted_indices = np.argsort(similarities[0])[::-1]
        
        # Retrieve the most similar drugs and ensure uniqueness
        unique_drugs = []
        for idx in sorted_indices:
            drug = combined_df.iloc[idx]["drug"]
            if drug not in unique_drugs:
                unique_drugs.append(drug)
            if len(unique_drugs) >= top_n:
                break
        
        return unique_drugs
    
    def compute_gradient_penalty(self, real_samples, fake_samples, disease_batch):
        alpha = torch.rand((real_samples.size(0), 1), device=self.device)
        alpha = alpha.expand(real_samples.size())
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(disease_batch, interpolates)
        fake = torch.ones(d_interpolates.size(), device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def evaluate_multiple_accuracy(self, df_eval, df_combined, disease_drug_map, device, top_n=5):
        correct_predictions = 0
        
        for _, row in df_eval.iterrows():
            disease_name, disease_embedding = row['disease'], row['disease_embedding']
            
            predicted_drugs = self.find_most_similar_drugs(
                disease_embedding,
                df_combined,
                device,
                top_n=top_n
            )
            
            # Check if any of the predicted drugs are in the set of valid drugs for this disease
            if any(predicted_drug in disease_drug_map[disease_name] for predicted_drug in predicted_drugs):
                correct_predictions += 1
                
        accuracy = correct_predictions / len(df_eval)
        return accuracy
    
    def train_step(self, disease_batch, drug_batch, disease_names):
        batch_size = disease_batch.size(0)
        real_labels = torch.ones((batch_size, 1), device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Real samples
        real_pred = self.discriminator(disease_batch, drug_batch)
        d_real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Generate fake samples
        generated_drugs = self.generator(disease_batch)
        generated_drugs = nn.functional.normalize(generated_drugs, dim=1)
        fake_pred = self.discriminator(disease_batch, generated_drugs.detach())
        d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        # Compute gradient penalty
        gradient_penalty = self.compute_gradient_penalty(drug_batch, generated_drugs, disease_batch)
        lambda_gp = 10  # Coefficient for the gradient penalty term
        
        # Total discriminator loss with gradient penalty
        d_loss = d_real_loss + d_fake_loss + lambda_gp * gradient_penalty
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.optimizer_D.step()
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        generated_drugs = self.generator(disease_batch)
        generated_drugs = nn.functional.normalize(generated_drugs, dim=1)
        
        # Adversarial loss
        fake_pred = self.discriminator(disease_batch, generated_drugs)
        g_loss = self.adversarial_loss(fake_pred, real_labels)
        
        # Initialize total triplet loss
        triplet_losses = []
        similarity_rewards = torch.zeros(batch_size, device=self.device)
        similarity_scores = torch.zeros(batch_size, device=self.device)
        
        for i, (gen_drug, disease_name) in enumerate(zip(generated_drugs, disease_names)):
            # Get positive samples (valid drugs for this disease)
            valid_drugs = self.disease_to_drugs[disease_name]
            
            # Get negative samples
            disease_indices = torch.tensor([
                i for i, d in enumerate(self.df['disease']) if d == disease_name
            ], dtype=torch.long, device=self.device)
            negative_mask = torch.ones(len(self.drug_embeddings), dtype=torch.bool, device=self.device)
            negative_mask[disease_indices] = False
            negative_drugs = self.drug_embeddings[negative_mask]
            
            # Sample hardest negative (most similar to generated)
            with torch.no_grad():
                neg_sims = torch.cosine_similarity(
                    gen_drug.unsqueeze(0),
                    negative_drugs
                )
                # Select top-k hardest negatives
                top_k_neg_indices = torch.topk(neg_sims, k=5, largest=True)[1]
                top_k_negatives = negative_drugs[top_k_neg_indices]
            
            # Sample random positive
            pos_idx = torch.randint(0, len(valid_drugs), (1,))
            positive = valid_drugs[pos_idx]
            
            # Compute triplet loss with averaged top-k negatives
            triplet_losses.append(
                self.triplet_loss(
                    gen_drug.unsqueeze(0),
                    positive,
                    top_k_negatives,  # Use top-k negatives
                    margin=0.3
                )
            )
            
            # Compute regular similarity reward
            reward, similarity = self.compute_similarity_reward(gen_drug, disease_name)
            similarity_rewards[i] = reward
            similarity_scores[i] = similarity
        
        triplet_loss = torch.stack(triplet_losses).mean()
        similarity_loss = -torch.mean(similarity_rewards)
        
        # Combined loss with weighted components
        g_total_loss = (
            1.0 * g_loss +           # adversarial loss
            2.0 * similarity_loss +  # contrastive loss
            1.0 * triplet_loss       # triplet loss
        )
        
        g_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer_G.step()
        
        return {
            'g_loss': g_total_loss.item(),
            'd_loss': d_loss.item(),
            'adversarial_loss': g_loss.item(),
            'similarity_mean': similarity_scores.mean().item(),
            'triplet_loss': triplet_loss.item()
        }
    
    def train(self, num_epochs=500, patience=50):
        from tqdm.auto import tqdm, trange
        
        disease_indices = torch.tensor(range(len(self.df)), dtype=torch.long)
        dataset = TensorDataset(
            self.disease_embeddings,
            self.drug_embeddings,
            disease_indices
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Create progress bar for epochs
        epoch_pbar = trange(num_epochs, desc='Training')
        
        for epoch in epoch_pbar:
            g_losses = []
            d_losses = []
            
            # Create progress bar for batches
            batch_pbar = tqdm(
                dataloader,
                desc=f'Epoch {epoch}',
                leave=False
            )
            
            for disease_batch, drug_batch, indices in batch_pbar:
                disease_names = [self.df['disease'].iloc[idx.item()] for idx in indices]
                
                metrics = self.train_step(
                    disease_batch,
                    drug_batch,
                    disease_names
                )
                
                g_losses.append(metrics['g_loss'])
                d_losses.append(metrics['d_loss'])
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'G_loss': f"{metrics['g_loss']:.4f}",
                    'D_loss': f"{metrics['d_loss']:.4f}",
                    'Adversarial_loss': f"{metrics['adversarial_loss']:.4f}",
                    'Sim': f"{metrics['similarity_mean']:.4f}",
                    'Triplet_loss': f"{metrics['triplet_loss']:.4f}"
                })
            
            # Update learning rates
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            
            # Compute test similarity
            test_similarity = self.evaluate_multiple_accuracy(
                self.df_test_filtered,
                self.df_test_filtered,
                self.disease_drug_map_test,
                device,
                top_n=5
            )
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'G_loss': f"{avg_g_loss:.4f}",
                'D_loss': f"{avg_d_loss:.4f}",
                'Test_Sim': f"{test_similarity:.4f}",
                'LR_G': f"{self.optimizer_G.param_groups[0]['lr']:.6f}",
                'LR_D': f"{self.optimizer_D.param_groups[0]['lr']:.6f}"
            })

            # Early stopping check
            if test_similarity > self.best_similarity:
                print(f"Updating the best model state at epoch {epoch}, model test similarity: {test_similarity:.4f}")
                self.best_similarity = test_similarity
                self.best_generator_state = copy.deepcopy(self.generator.state_dict())
                self.stagnant_epochs = 0
            else:
                self.stagnant_epochs += 1

            # Early stopping condition
            if self.stagnant_epochs >= patience:
                print(f"\nStopping early at epoch {epoch} due to no improvement in test similarity.")
                if self.best_generator_state is not None:
                    self.generator.load_state_dict(self.best_generator_state)
                    restored_similarity = self.evaluate_multiple_accuracy(
                        self.df_test_filtered,
                        self.df_test_filtered,
                        self.disease_drug_map_test,
                        device,
                        top_n=5
                    )
                    print(f"Restored model test similarity: {restored_similarity:.4f} (Expected: {self.best_similarity:.4f})")
                break


    def generate_drug(self, disease_embedding):
        if not isinstance(disease_embedding, torch.Tensor):
            disease_embedding = torch.tensor(
                disease_embedding, 
                dtype=torch.float32
            ).to(self.device)
        
        if len(disease_embedding.shape) == 1:
            disease_embedding = disease_embedding.unsqueeze(0)
        
        # Normalize input
        disease_embedding = nn.functional.normalize(disease_embedding, dim=1)
        
        self.generator.eval()
        with torch.no_grad():
            generated_drug = self.generator(disease_embedding)
            generated_drug = nn.functional.normalize(generated_drug, dim=1)
        self.generator.train()
        
        return generated_drug

model = DrugGAN(df, df_test_filtered, batch_size=32)
model.train(num_epochs=500)

torch.save(model.generator.state_dict(), 'models/generator_63_17.pth')
"""Streamlit demo for fashion item generation."""

import os
import streamlit as st
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.fashion_generation.sampling import FashionSampler, load_model_for_sampling
from src.fashion_generation.utils import get_device


def load_model_and_sampler():
    """Load model and create sampler."""
    if "model" not in st.session_state:
        # Load configuration
        config = OmegaConf.load("configs/config.yaml")
        
        # Setup device
        device = get_device(config.device)
        
        # Load model (you'll need to provide a checkpoint path)
        checkpoint_path = st.selectbox(
            "Select model checkpoint:",
            [f for f in os.listdir("checkpoints") if f.endswith(".ckpt")]
        )
        
        if checkpoint_path:
            model, device = load_model_for_sampling(
                os.path.join("checkpoints", checkpoint_path),
                config.model,
                device
            )
            
            sampler = FashionSampler(model, device, config.sampling)
            
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.sampler = sampler


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Fashion Item Generation",
        page_icon="👗",
        layout="wide"
    )
    
    st.title("👗 Fashion Item Generation Demo")
    st.markdown("Generate new fashion items using a trained GAN model!")
    
    # Load model
    load_model_and_sampler()
    
    if "sampler" not in st.session_state:
        st.warning("Please select a model checkpoint to continue.")
        return
    
    sampler = st.session_state.sampler
    
    # Sidebar controls
    st.sidebar.header("Generation Controls")
    
    # Number of samples
    num_samples = st.sidebar.slider(
        "Number of samples:",
        min_value=1,
        max_value=100,
        value=16,
        step=1
    )
    
    # Grid size
    grid_size = st.sidebar.slider(
        "Grid size:",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )
    
    # Random seed
    seed = st.sidebar.number_input(
        "Random seed (optional):",
        min_value=0,
        max_value=1000000,
        value=None,
        step=1
    )
    
    # Generation type
    generation_type = st.sidebar.selectbox(
        "Generation type:",
        ["Random samples", "Interpolation"]
    )
    
    # Generate button
    if st.sidebar.button("Generate Samples", type="primary"):
        with st.spinner("Generating samples..."):
            if generation_type == "Random samples":
                # Generate random samples
                samples = sampler.generate_samples(
                    num_samples=num_samples,
                    seed=seed
                )
                
                # Create grid
                grid = sampler.generate_grid(
                    grid_size=grid_size,
                    seed=seed
                )
                
                # Display grid
                st.subheader("Generated Fashion Items")
                
                # Convert tensor to numpy for display
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(grid_np)
                ax.set_title("Generated Fashion Items")
                ax.axis('off')
                st.pyplot(fig)
                
            else:  # Interpolation
                # Generate two random latent vectors
                z1 = torch.randn(1, sampler.model.generator.z_dim, device=sampler.device)
                z2 = torch.randn(1, sampler.model.generator.z_dim, device=sampler.device)
                
                # Interpolate
                interpolated = sampler.interpolate(
                    z1, z2,
                    num_steps=num_samples
                )
                
                # Display interpolation
                st.subheader("Latent Space Interpolation")
                
                # Create grid for interpolation
                grid = sampler.generate_grid(
                    grid_size=grid_size,
                    seed=seed
                )
                
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.imshow(grid_np)
                ax.set_title("Latent Space Interpolation")
                ax.axis('off')
                st.pyplot(fig)
    
    # Model information
    st.sidebar.header("Model Information")
    
    if "model" in st.session_state:
        model = st.session_state.model
        
        from src.fashion_generation.utils import get_model_size
        
        st.sidebar.metric("Generator Parameters", get_model_size(model.generator))
        st.sidebar.metric("Discriminator Parameters", get_model_size(model.discriminator))
        st.sidebar.metric("Device", str(st.session_state.device))
    
    # Instructions
    st.markdown("""
    ## Instructions
    
    1. **Select a model checkpoint** from the dropdown above
    2. **Adjust generation parameters** in the sidebar:
       - Number of samples: How many images to generate
       - Grid size: Size of the display grid
       - Random seed: For reproducible results
    3. **Choose generation type**:
       - Random samples: Generate random fashion items
       - Interpolation: Show smooth transition between two random items
    4. **Click "Generate Samples"** to create new fashion items!
    
    ## About
    
    This demo uses a DCGAN (Deep Convolutional Generative Adversarial Network) 
    trained on the Fashion-MNIST dataset to generate new fashion item designs.
    
    The model learns to create realistic-looking clothing items including:
    - T-shirts/tops
    - Trousers
    - Pullovers
    - Dresses
    - Coats
    - Sandals
    - Shirts
    - Sneakers
    - Bags
    - Ankle boots
    """)


if __name__ == "__main__":
    main()

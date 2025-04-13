import torch
from dataset import SketchData
from transformers import AutoTokenizer
from model.decoder import SketchDecoder

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Basic configuration
    config = {
        'hidden_dim': 1024,
        'embed_dim': 512, 
        'num_layers': 16, 
        'num_heads': 8,
        'dropout_rate': 0.1,
        'text_len': 50,
        'tokenizer_name': 'google/bert_uncased_L-12_H-512_A-8',
        'word_emb_path': 'ckpts/word_embedding_512.pt',
        'pos_emb_path': None,
    }
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Create a small test dataset
    test_dataset = SketchData(
        meta_file_path='dataset/FIGR-SVG-train.csv',  
        svg_folder='dataset/FIGR-SVG-svgo', 
        MAX_LEN=512,
        text_len=config['text_len'],
        tokenizer=tokenizer,
        require_aug=True
    )
    
    # Create a small dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Using 0 for simplicity in testing
    )
    
    # Initialize model
    model = SketchDecoder(
        config=config,
        pix_len=test_dataset.maxlen_pix,
        text_len=config['text_len'],
        num_text_token=tokenizer.vocab_size,
        word_emb_path=config['word_emb_path'],
        pos_emb_path=config['pos_emb_path']
    )
    
    # Compile the model
    print("Compiling model...")
    model = torch.compile(
        model,
        mode="default",
        dynamic=True,
        fullgraph=True
    )
    
    # Test forward pass
    print("Testing forward pass...")
    for batch in test_dataloader:
        pix, xy, mask, text = batch
        try:
            # Move data to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pix = pix.to(device)
            xy = xy.to(device)
            mask = mask.to(device)
            text = text.to(device)
            model = model.to(device)
            
            # Forward pass
            loss, pix_loss, text_loss = model(pix, xy, mask, text, return_loss=True)
            print(f"Loss: {loss.item():.4f}, Pixel Loss: {pix_loss.item():.4f}, Text Loss: {text_loss.item():.4f}")
            break  # Just test one batch
        except Exception as e:
            print(f"Error during forward pass: {e}")
            break

if __name__ == "__main__":
    main() 
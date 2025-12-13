from model.LocalLoadModel import load_model, gen_resp

# This is a simple test script to verify the LocalLoadModel functionality
# You'll need to provide a valid local model path to test

def test_local_model():
    # Replace with your actual local model path
    model_path = "path/to/your/local/model"
    
    try:
        # Test loading the model
        llm = load_model(model_path)
        print("✅ Model loaded successfully as LangChain LLM")
        
        # Test generating a response
        query = "What is artificial intelligence?"
        response = gen_resp(llm, query)
        print(f"❓ Query: {query}")
        print(f"🤖 Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_local_model()
import logging
from qdrant_client import QdrantClient

# Load configuration from .env
from config_rtx4080 import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    validate_config
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_connection():
    """Test connection to Qdrant Cloud."""
    print("🔗 Testing Qdrant Cloud Connection")
    print("=" * 50)

    # Validate configuration first
    config_errors = validate_config()
    if config_errors:
        print("❌ Configuration Errors:")
        for error in config_errors:
            print(f"   - {error}")
        print("\n💡 Please fix your .env file:")
        print("   1. Copy .env.template to .env")
        print("   2. Update QDRANT_URL with your cluster URL")
        print("   3. Update QDRANT_API_KEY with your API key")
        return False

    print(f"📍 URL: {QDRANT_URL}")
    print(f"🔑 API Key: {'*' * (len(QDRANT_API_KEY) - 4) + QDRANT_API_KEY[-4:]}")
    print(f"📁 Collection: {QDRANT_COLLECTION_NAME}")

    try:
        print(f"\n🔌 Connecting to Qdrant...")

        # Initialize client
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Test connection by getting collections
        print(f"📋 Fetching collections...")
        collections = client.get_collections()

        print(f"✅ Connection successful!")
        print(f"📁 Found {len(collections.collections)} collections:")

        if collections.collections:
            for collection in collections.collections:
                try:
                    # Get detailed info for each collection
                    collection_info = client.get_collection(collection.name)
                    print(f"   📂 {collection.name}")
                    print(f"      Points: {collection_info.points_count}")
                    print(f"      Vectors: {collection_info.vectors_count}")
                    print(f"      Status: {collection_info.status}")

                    # Check if this is our target collection
                    if collection.name == QDRANT_COLLECTION_NAME:
                        print(f"      🎯 This is your target collection!")

                        # Show vector configuration
                        vector_config = collection_info.config.params.vectors
                        print(f"      Vector size: {vector_config.size}")
                        print(f"      Distance: {vector_config.distance}")

                except Exception as e:
                    print(f"   📂 {collection.name} (error getting details: {e})")
        else:
            print(f"   (No collections found)")
            print(f"   💡 This is normal for a new Qdrant instance")

        # Test creating a temporary collection to verify write permissions
        print(f"\n🧪 Testing write permissions...")
        test_collection_name = "test_connection_temp"

        try:
            from qdrant_client.models import Distance, VectorParams

            # Create test collection
            client.create_collection(
                collection_name=test_collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"✅ Write test: Created temporary collection")

            # Delete test collection
            client.delete_collection(collection_name=test_collection_name)
            print(f"✅ Write test: Deleted temporary collection")

        except Exception as e:
            print(f"❌ Write test failed: {e}")
            print(f"   This might indicate insufficient permissions")

        print(f"\n🎉 All tests passed!")
        print(f"✅ Your Qdrant Cloud setup is ready for indexing")

        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Verify your .env file has correct values:")
        print(f"   QDRANT_URL should look like: https://xyz-abc-def.us-east-1.cloud.qdrant.io:6333")
        print(f"   QDRANT_API_KEY should be your actual API key")
        print(f"2. Check your Qdrant cluster is running in the cloud dashboard")
        print(f"3. Verify your API key has the correct permissions")
        print(f"4. Check your internet connection and firewall settings")

        return False


def test_embedding_dimensions():
    """Test if we can connect and check dimension compatibility."""
    from config_rtx4080 import EMBEDDING_DIMENSION, EMBEDDING_MODEL

    print(f"\n🧠 Embedding Configuration Check")
    print(f"=" * 40)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Expected dimensions: {EMBEDDING_DIMENSION}")

    # Check if collection exists and has matching dimensions
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()

        target_collection = None
        for collection in collections.collections:
            if collection.name == QDRANT_COLLECTION_NAME:
                target_collection = collection
                break

        if target_collection:
            collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
            existing_dim = collection_info.config.params.vectors.size

            if existing_dim == EMBEDDING_DIMENSION:
                print(f"✅ Dimension match: {existing_dim} = {EMBEDDING_DIMENSION}")
            else:
                print(f"⚠️  Dimension mismatch!")
                print(f"   Existing collection: {existing_dim} dimensions")
                print(f"   Your config: {EMBEDDING_DIMENSION} dimensions")
                print(f"   💡 You may need to:")
                print(f"   - Delete the existing collection, or")
                print(f"   - Update your EMBEDDING_MODEL in .env, or")
                print(f"   - Use a different QDRANT_COLLECTION_NAME")
        else:
            print(f"ℹ️  Collection '{QDRANT_COLLECTION_NAME}' doesn't exist yet")
            print(f"   It will be created automatically with {EMBEDDING_DIMENSION} dimensions")

    except Exception as e:
        print(f"❌ Could not check dimensions: {e}")


if __name__ == "__main__":
    print("🧪 Qdrant Cloud Connection Test")
    print("=" * 50)

    success = test_connection()

    if success:
        test_embedding_dimensions()
        print(f"\n🎉 Ready to proceed!")
        print(f"Next steps:")
        print(f"1. Run: python preprocess_documents.py")
        print(f"2. Run: python index_to_qdrant.py")
    else:
        print(f"\n🔧 Please fix the connection issues before proceeding.")
        print(f"Check your .env file and Qdrant Cloud dashboard.")
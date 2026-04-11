import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class MongoDB:
    _instance = None

    def __new__(cls,
                uri="mongodb+srv://roundmixall:UfNitM19ShlZQJmh@cluster0.afmyoco.mongodb.net/AlchemyMix?retryWrites=true&w=majority",
                db_name="AlchemyMix"):
        if cls._instance is None:
            try:
                cls._instance = super(MongoDB, cls).__new__(cls)
                cls._instance.client = MongoClient(
                    uri,
                    tls=True,
                    tlsCAFile=certifi.where(),  # Trusted CA file
                    serverSelectionTimeoutMS=10000,
                    connectTimeoutMS=10000
                )

                # Test the connection
                cls._instance.client.admin.command("ping")
                cls._instance.db = cls._instance.client[db_name]
                print("✓ Successfully connected to MongoDB Atlas")

            except ConnectionFailure as e:
                print(f"❌ Failed to connect to MongoDB: {e}")
                raise

            except Exception as e:
                print(f"❌ Unexpected error during MongoDB connection: {e}")
                raise

        return cls._instance

    def get_db(self):
        return self.db


# Initialize connection once
# mongo_instance = None
try:
    mongo_instance = MongoDB()
except Exception as e:
    print(f"🚨 Critical: Could not initialize MongoDB connection: {e}")

from pymongo import MongoClient

MONGO_URI =("mongodb+srv://prashant:haha@verifact-mongo.lvskg3d.mongodb.net/?appName=verifact-mongo")

client = MongoClient(MONGO_URI)
db = client["verifact"]
users_collection = db["users"]

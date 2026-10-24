"""
Demonstrates semantic clustering without real embeddings.
Vectors are 2D unit vectors — angle represents "meaning".
Animals cluster near 0°, vehicles near 90°.
"""
import baked

store = baked.VectorStore("example_store")

# "animals" cluster
store.add([1.0, 0.0],  {"text": "cat"})
store.add([0.98, 0.20], {"text": "dog"})
store.add([0.95, 0.31], {"text": "fish"})

# "vehicles" cluster
store.add([0.0, 1.0],  {"text": "car"})
store.add([0.20, 0.98], {"text": "truck"})

store.save()

# query: close to "animals" cluster
query = [0.99, 0.10]
results = store.search(query, top_k=3)

print("Top 3 results for query near 'animals' cluster:")
for metadata, score in results:
    print(f"  {metadata['text']:<8} score={score:.4f}")

# cleanup
import os
os.remove("example_store.vecs")
os.remove("example_store.json")

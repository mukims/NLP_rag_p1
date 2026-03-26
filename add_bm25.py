import json

def add_bm25_cells():
    with open("test.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### BM25 Index Integration\n",
            "This cell tokenizes the texts array and creates a keyword search index using `rank_bm25`."
        ]
    }

    code_source = [
        "from rank_bm25 import BM25Okapi\n",
        "import pickle\n",
        "\n",
        "# 1. Tokenize the texts. A simple lowercase and split() matches standard Okapi.\n",
        "tokenized_corpus = [doc.lower().split(\" \") for doc in texts]\n",
        "\n",
        "# 2. Fit the BM25 model\n",
        "bm25 = BM25Okapi(tokenized_corpus)\n",
        "\n",
        "# 3. Save the model to disk so our retriever can find it mapping 1:1 to ChromaDB\n",
        "with open(\"bm25_index.pkl\", \"wb\") as f:\n",
        "    pickle.dump(bm25, f)\n",
        "\n",
        "print(f\"BM25 index created for {len(texts)} documents and saved to bm25_index.pkl!\")\n"
    ]

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_source
    }

    # Remove any empty trailing cells first for cleanliness
    while nb["cells"] and nb["cells"][-1]["source"] == []:
        nb["cells"].pop()

    nb["cells"].extend([md_cell, code_cell])

    with open("test.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    add_bm25_cells()

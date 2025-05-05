from langchain_community.document_loaders import ArxivLoader

def get_arxiv_content_and_summary_from_title(title: str):
    loader = ArxivLoader(
        query=title,
        load_max_docs=1,  # max number of documents
        load_all_available_meta=True,  # load all available metadata
    )
    docs = loader.load()
    return docs[-1]
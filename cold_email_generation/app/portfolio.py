import pandas as pd
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path="resource/my_portfolio.csv"):
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name='portfolio')
    

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row['Techstack'] ,
                                    metadatas ={"links": row["Links"]},
                                    ids =[str(uuid.uuid4())])
                
    # def query_links(self , skills):
    #     return self.collection.query(query_texts=[skills] , n_results=2).get('metadatas',[])
    
    def query_links(self, skills):
        if isinstance(skills, list):
            skills_text = " ".join(skills)
        else:
            skills_text = str(skills)

        results = self.collection.query(
            query_texts=[skills_text],
            n_results=2
        )
        return results.get('metadatas', [])



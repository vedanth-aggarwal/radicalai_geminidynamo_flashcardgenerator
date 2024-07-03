from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import json
import logging
import re
# import ast
# from pydantic import BaseModel, Field
# from typing import Dict
# import json
# import ast
# import logging
# from langchain.prompts import PromptTemplate
# from langchain.output_parsers import JsonOutputParser
# from langchain.chains import SimpleChain
# from pydantic import BaseModel, Field
# from typing import Dict

# Configure log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiProcessor:
    def __init__(self, model_name, project):
        self.model = VertexAI(model_name=model_name, project=project)
    def generate_document_summary(self, documents: list, **args):
        
        chain_type = "map_reduce" if len(documents) > 10 else "stuff"
        
        chain = load_summarize_chain(
            llm = self.model,
            chain_type = chain_type,
            **args
        )
        
        return chain.run(documents)
    
    def count_total_tokens(self, docs: list):
        temp_model = GenerativeModel("gemini-1.0-pro",)
        total = 0
        logger.info("Counting total billable characters...")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
        return total
        
    
    def get_model(self):
        return self.model

class YoutubeProcessor:
    # Retrieve the full transcript
    
    def __init__(self, genai_processor: GeminiProcessor,parser):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 0
        )
        self.GeminiProcessor = genai_processor
        self.parser = parser
    def retrieve_youtube_documents(self, video_url: str, verbose = False):
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)
        print(result[:2])
        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)
        
        if verbose:
            total_billable_characters = self.GeminiProcessor.count_total_tokens(result)
            logging.info(f"{author}\n{length}\n{title}\n{total_size}\n{total_billable_characters}")
        
        return result
    def clean_json_string(json_str):
                """Clean JSON String capturing only the value between curly braces
                Args:
                    json_str (str): uncleaned string
                Returns:
                    str: cleaned string
                """
                # Define a regex pattern to match everything before and after the curly braces
                pattern = r'^.*?({.*}).*$'
                # Use re.findall to extract the JSON part from the string
                matches = re.findall(pattern, json_str, re.DOTALL)
                if matches:
                    # If there's a match, return the first one (should be the JSON)
                    return matches[0]
                else:
                    # If no match is found, return None
                    return None
                
    def find_key_concepts(self, documents:list, sample_size: int = 0, verbose = False):
        # iterate through all documents of group size N and find key concepts
        if sample_size > len(documents):
            raise ValueError("Group size is larger than the number of documents")
        
        # Optimize sample size given no input
        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose: logging.info(f"No sample size specified. Setting number of documents per sample as 5. Sample Size: {sample_size}")

        # Find number of documents in each group
        num_docs_per_group = len(documents) // sample_size + (len(documents) % sample_size > 0)
        
        # Check thresholds for response quality 
        if num_docs_per_group > 10:
            raise ValueError("Each group has more than 10 documents and output quality will be degraded significantly. Increase the sample_size parameter to reduce the number of documents per group.")
        elif num_docs_per_group > 5:
            logging.warn("Each group has more than 5 documents and output quality is likely to be degraded. Consider increasing the sample size.")
        
        # Split the document in chunks of size num_docs_per_group
        groups = [documents[i:i+num_docs_per_group] for i in range(0, len(documents), num_docs_per_group)]
        
        batch_concepts = []
        batch_cost = 0
        
        logger.info("Finding key concepts...")
        for group in tqdm(groups):
            # Combine content of documents per group
            group_content = ""
            
            for doc in group:
                group_content += doc.page_content
            
            # Prompt for finding concepts 
            prompt2 = PromptTemplate(
                template = """
                Find and define key concepts or terms found in the text:
                {text}
                
                Respond in the following format as a JSON object without any backticks separating each concept with a comma.Strictly follow the format given:
                {{"concept": "definition", "concept": "definition", ...}}
                """,
                input_variables=["text"]
            )
            
            prompt = PromptTemplate(
                template="""
                Find and define key concepts or terms found in the text:
                {text}
                
                Respond in the following format as a JSON object without any backticks separating each concept with a comma. Strictly follow the format given:
                {format_instructions}
                """,
                input_variables=["text"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )
            
            # Create chain
            chain = prompt | self.GeminiProcessor.model
            # str(data).strip("'<>() ").replace('\'','\"')
            # Run chain
            output_concept = chain.invoke({"text": group_content})
            # output_concept = clean_json_string(output_concept)
            batch_concepts.append(output_concept)
            
            # Post Processing Observation
            if verbose:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char/1000) * 0.000125
                
                logging.info(f"Running chain on {len(group)} documents")
                logging.info(f"Total input characters: {total_input_char}")
                logging.info(f"Total cost: {total_input_cost}")
                
                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000) * 0.000375
                
                logging.info(f"Total output characters: {total_output_char}")
                logging.info(f"Total cost: {total_output_cost}")
                
                batch_cost += total_input_cost + total_output_cost
                logging.info(f"Total group cost: {total_input_cost + total_output_cost}\n")
            
        # Convert each JSON string in batch_concepts to a Python Dict
        #print(len(batch_concepts))
        #for concept in batch_concepts:
        #    print(concept,type(concept))
        #    print(json.loads(concept))
        #    print('\n\n')
        #processed_concepts = [ast.literal_eval(concept) for concept in batch_concepts]
        #[ast.literal_eval(concept) for concept in batch_concepts] #[json.loads(concept) for concept in batch_concepts]
        processed_concepts = [self.parser.parse(concept) for concept in batch_concepts]
        print(processed_concepts)
        logging.info(f"Total Analysis Cost: ${batch_cost}")    
        return processed_concepts
    
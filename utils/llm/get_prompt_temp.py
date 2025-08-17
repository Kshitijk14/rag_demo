from langchain.prompts import PromptTemplate


prompt_retrieval_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It dos not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'YES' or 'NO' score to indicate whether the document is relevant to the question or not. \n 
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document}\n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

prompt_generate_answer = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise and to the point.
    Answer the question using the context below.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} \n
    Context: {context} \n
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)
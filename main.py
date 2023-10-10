from bs4 import BeautifulSoup

import os
import sys
import asyncio
import unicodedata
import pandas as pd
from chatgpt_core import GPTCore

## --- CONFIGURABLE CONSTANTS ---
## Generally speaking, these won't change.
CHUNK_SEPARATOR = "\n-----\n"
MAX_WORDS_PER_CHUNK = 10000
CHUNK_VARIABILITY_BUFFER = 100 ## dividing will not be precise, adding some buffer to prevent a chunk that's like...1 or 2 comments, which can't really be analyzed
GPT_MODEL_NAME = "gpt-3.5-turbo-16k"

## --- INSTRUCTIONS AND MESSAGES ---
## Tune these to your preference.

## This sets the disposition of ChatGPT and what rules it should follow.
INITIAL_INSTRUCTION = ("You are a psychological profiler analyzing Reddit comments. "
                       "Provide an unbiased and detailed profile, covering both positive and negative aspects. "
                       "Use the data available to the best of your ability.")

## This is the instruction for the chunk step, where it analyzes big blocks of comments.
CHUNK_INSTRUCTION = ('''Analyze the comments from redditor _USERNAME_. Provide a detailed profile based on the comments, 
                       categorizing your observations into:
                       - Communication Style
                       - Personality Traits & Attitudes
                       - Interests & Hobbies
                       - Political Ideology
                       - Values and Beliefs
                       - Other Notes
                       (If data is insufficient for a category, skip it.)''')


## This is a FAKE instruction, used in the final synthesis step, to make it think that it already did this. You probably should leave this as-is. (Basically you're faking this part of a conversation and we're putting all of the analysis as 'its reply' that the synthesis step will then work with)
SYNTHESIS_SETUP_INSTRUCTION = "Analyze all of the comments for redditor _USERNAME_. For each set of comments analyzed, produce a psychological profile of the user, separating each profile with \"-----\"."  

SYNTHESIS_EXECUTION_INSTRUCTION = ('''Consolidate the analyses into a single, comprehensive profile for redditor _USERNAME_. 
                                      Categorize your observations into:
                                      - Communication Style
                                      - Personality Traits & Attitudes
                                      - Interests & Hobbies
                                      - Political Ideology
                                      - Values and Beliefs
                                      - Other Notes
                                      (If data is insufficient for a category, skip it.)''')


## --- FUNCTIONS ---

def parse_html_file(file_path, output_file_path):

    """
    
    Parses the HTML file and extracts the comments data.\n

    Parameters:\n
    file_path (str) : The path to the HTML file.\n
    output_file_path (str) : The path to the CSV file to save the comments data to.\n

    Returns:\n
    df (DataFrame) : The comments data as a DataFrame.\n
    total_word_count (int) : The total word count of all comments.\n

    """

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    comments_data = []
    total_word_count = 0

    for tr in soup.find('table', {'id': 'resulttable'}).find_all('tr', style=""): ## type: ignore (we know it's not None)

        td = tr.find('td')
        if(td is None):
            continue

        h4 = td.find('h4')
        if(h4 is None):
            continue

        title = h4.get_text(strip=True)

        div_md = td.find('div', {'class': 'md'})
        if(div_md is None):
            continue
        
        comment_text_elements = []
        for element in div_md.recursiveChildGenerator():

            if(element.name == 'blockquote'):
                comment_text_elements.append('>')

            elif(element.name == 'p'):
                comment_text_elements.append('/n' + element.get_text())

            elif(element.name == 'a'):
                comment_text_elements.append(element.get_text())

        ## Join the elements and remove the first "/n" prefix
        comment_text = ''.join(comment_text_elements).lstrip('/n')

        ## Normalize unicode characters
        comment_text = unicodedata.normalize('NFKD', comment_text).encode('ascii', 'ignore').decode('utf-8')
        comments_data.append({'post_title': title, 'reply_comment': comment_text})

        ## Increment the total word count by the word count of the current comment
        total_word_count += len(comment_text.split())

    ## Create a DataFrame
    df = pd.DataFrame(comments_data)
    
    ## Save to CSV
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    
    return df, total_word_count

def break_into_chunks(comments_df, max_words):

    """

    Breaks the comments into chunks of a maximum word count.\n

    Parameters:\n
    comments_df (DataFrame) : The comments data as a DataFrame.\n
    max_words (int) : The maximum word count for each chunk.\n

    Returns:\n
    chunks_metadata_df (DataFrame) : The chunks metadata as a DataFrame.\n

    """

    word_count = 0
    start_index = 0
    chunks = []
    index = 0

    for index, row in comments_df.iterrows():
        comment_word_count = len(row['reply_comment'].split())

        if(word_count + comment_word_count <= max_words):
            word_count += comment_word_count
        else:
            chunks.append((start_index, index-1))
            start_index = index
            word_count = comment_word_count

    chunks.append((start_index, index))

    chunks_metadata_df = pd.DataFrame(chunks, columns=['from_line', 'to_line'])
    
    return chunks_metadata_df

##-------------------start-of-send_chunks_to_chatgpt()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

async def send_chunks_to_chatgpt(comments_df, chunks_metadata_df, model,username):

    """

    Sends the chunks to ChatGPT and returns the results.\n

    Parameters:\n
    comments_df (DataFrame) : The comments data as a DataFrame.\n

    Returns:\n
    results_df (DataFrame) : The results as a DataFrame.\n

    """

    tasks = []

    for index, row in chunks_metadata_df.iterrows():
        task = asyncio.create_task(process_chunk(index, row, comments_df, model, username))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    
    # Sort the results by the index
    sorted_results = sorted(results, key=lambda x: x[0])  # Assuming the index is the first element in the result tuple
    results_df = pd.DataFrame(sorted_results, columns=['from_line', 'to_line', 'response'])
    
    return results_df

##-------------------start-of-process_chunk()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

async def process_chunk(index, row, comments_df, model, username):

    """
    
    Processes a chunk of comments.\n

    Parameters:\n
    index (int) : The index of the chunk.\n

    Returns:\n
    [start_line, end_line, response] (list) : The start line, end line, and response.\n

    """

    start_line, end_line = row['from_line'], row['to_line']
    chunk_comments = comments_df.loc[start_line:end_line, 'reply_comment']
    compiled_comments = "\n-----\n".join(chunk_comments)

    GPT_CORE = GPTCore(instructions=INITIAL_INSTRUCTION, model=model)
    GPT_CORE.add_message(compiled_comments, actor="user")
    GPT_CORE.add_message(CHUNK_INSTRUCTION.replace('_USERNAME_', username), actor="user")
    response = await GPT_CORE.generate_response()

    return index, start_line, end_line, response

def synthesize_profiles(username, results_df, model):

    """

    Synthesizes the profiles into a comprehensive report.\n

    Parameters:\n
    username (str) : The username of the redditor.\n
    results_df (DataFrame) : The results as a DataFrame.\n
    model (str) : The GPT model to use.\n

    Returns:\n
    synthesized_response (str) : The synthesized response.\n

    """

    ## Initializing the GPTCore instance with the new instructions
    GPT_CORE = GPTCore(instructions=INITIAL_INSTRUCTION, model=model)
    
    ## Combining all GPT responses into a single message, in reverse order
    combined_message = "\n-----\n".join(results_df['response'][::-1])
    
    ## Adding the combined message to GPT_CORE, with instructions
    GPT_CORE.add_message(SYNTHESIS_SETUP_INSTRUCTION.replace('_USERNAME_', username),actor="user")
    GPT_CORE.add_message(combined_message,actor="assistant")
    GPT_CORE.add_message(SYNTHESIS_EXECUTION_INSTRUCTION.replace('_USERNAME_', username),actor="user")
    
    ## Generating the synthesized response from ChatGPT
    synthesized_response = GPT_CORE.generate_response()
    
    return synthesized_response

##-------------------start-of-save_to_file()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def save_to_file(username, content):

    """
    
    Saves the synthesized profile to a file.\n

    Parameters:\n
    username (str) : The username of the redditor.\n
    content (str) : The content to save.\n

    Returns:\n
    None.\n

    """

    with open(f"{username}_synthesized_profile.txt", "w") as file:
        file.write(content)

##-------------------start-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if(__name__ == "__main__"):
    username = sys.argv[1]
    html_file_path = f"{username}.html"
    csv_file_path = f"{username}_comments_data.csv"
    gpt_response_csv_path = f'{username}_gpt_responses.csv'

    comments_df, total_word_count = parse_html_file(html_file_path, csv_file_path)
    
    ## Calculate the optimal chunk size
    chunk_size = total_word_count
    num_chunks = 1

    while(chunk_size / num_chunks > MAX_WORDS_PER_CHUNK):
        num_chunks += 1
    chunk_size = (chunk_size // num_chunks) + CHUNK_VARIABILITY_BUFFER
    
    print(f"Breaking into chunks of word count: {chunk_size}")
    chunks_metadata_df = break_into_chunks(comments_df, chunk_size)
    print(chunks_metadata_df)

    if(os.path.exists(gpt_response_csv_path)):
        results_df = pd.read_csv(gpt_response_csv_path)
    else:
        results_df = asyncio.run(send_chunks_to_chatgpt(comments_df, chunks_metadata_df, GPT_MODEL_NAME, username))
        results_df.to_csv(gpt_response_csv_path, index=False)

    if(len(chunks_metadata_df) <= 1):
        # If there is only one chunk, just save the response to a file, as it was already printed to console
        content = results_df.iloc[0]['response']
        save_to_file(username, content)
        
    else:
        ## Synthesizing the profiles into a comprehensive report
        synthesized_profile = synthesize_profiles(username, results_df, GPT_MODEL_NAME)
        
        ## Printing the synthesized profile to console
        print(synthesized_profile)
        
        ## Saving the synthesized profile to a file
        save_to_file(username, synthesized_profile)
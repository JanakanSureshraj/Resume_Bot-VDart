import httpx
import ollama
import time

def score_applicant_responses(questions, answers):
    try:
        # Concatenate questions and answers
        input_text = ""
        for question, answer in zip(questions, answers):
            input_text += f"Q: {question}\nA: {answer}\n\n"
        
        response = ollama.chat(model='llama3', messages=[
            {
                "role": "user", 
                "content": 'The following text contains questions asked to an applicant and the applicant\'s transcribed answers. I want you to provide a score from 0 to 100 for the quality of the applicant\'s answers. Do a thorough analysis before coming up with your final scores. Provide a single score in the format: "Score: X". Nothing else should be output. Here is the data: ' + input_text,
            },
        ])
        return response['message']['content']
    except httpx.RequestError as e:
        print(f"An HTTP request error occurred: {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"An HTTP status error occurred: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while scoring the responses: {e}")
        return None

if __name__ == "__main__":
    # Example input
    questions = [
        "What is your experience with data analysis?",
        "Can you describe a project where you used machine learning?",
        "How do you handle tight deadlines?"
    ]
    answers = [
        "I have over 3 years of experience in data analysis, working with tools like SQL, Python, and Excel to analyze and interpret data.",
        "In my recent project, I developed a machine learning model to predict customer churn using logistic regression. This helped the company reduce churn by 15%.",
        "I prioritize tasks, break down large projects into smaller tasks, and use tools like Trello to manage my workflow effectively to meet deadlines."
    ]
    
    start_time = time.time()
    
    score = score_applicant_responses(questions, answers)
    if score:
        print(f"Applicant's Answer Quality Score: {score}")
    else:
        print("Failed to score the responses.")
    
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")
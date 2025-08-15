from openai import OpenAI

client = OpenAI(api_key="API KEY HERE")

def llm_judge(question: str, model_response: str, ground_truth: str) -> str:
    """
    Use an LLM as a judge to determine if the model response matches the ground truth.
    Returns "success" or "fail".
    
    Args:
        question (str): The original question.
        model_response (str): The model's answer to the question.
        ground_truth (str): The correct answer for the question.
    """
    
    prompt = f"""
                You are an impartial judge. Given a question, a model's response, and the gold-standard answer, 
                determine whether the model's response is correct. Return only "success" if correct, or "fail" if incorrect.

                Comparison rules:
                - Ignore case and whitespace.
                - Consider numbers equal if they match within 0.01.
                - Treat equivalent expressions (e.g., "16th place" = "last place") as correct.
                - Do not include any explanation in your output, only "success" or "fail".

                Question: {question}
                Model Response: {model_response}
                Ground Truth: {ground_truth}

                Judge:
                """

    completion = client.chat.completions.create(
        model="gpt-4o", # might wanna test which is best here for us
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Extract and clean judge decision
    decision = completion.choices[0].message.content.strip().lower()

    if decision not in ["success", "fail"]:
        raise ValueError(f"Unexpected judge output: {decision}")

    return decision


if __name__ == "__main__":
    # Example usage
    result = llm_judge(
        question="What score did this individual receive for this event?",
        model_response="10.50",
        ground_truth="10.50"
    )
    print("Judge result:", result)

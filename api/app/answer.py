from .graph import run_questionnaire_workflow


def answer_question(question: str) -> dict:
    """Answers security questionnaire questions using the LangGraph self-checking workflow."""
    return run_questionnaire_workflow(question)

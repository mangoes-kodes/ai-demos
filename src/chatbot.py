from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

def demo_chatbot():
    demo_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    demo_memory = ConversationSummaryBufferMemory(llm=demo_llm, max_token_limit=1000)
    demo_conversation = ConversationChain(llm=demo_llm, memory=demo_memory)
    print("Welcome to the chatbot demo! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = demo_conversation.run(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    demo_chatbot()

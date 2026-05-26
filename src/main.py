
from dotenv import load_dotenv
load_dotenv()

from utils.graph_builder import graph

if __name__ == "__main__":
    app = graph()
    
    target_website = input("Enter the URL of the company you want to analyze: ") 
    
    initial_state = {
        "target_url": target_website,
        "target_info": "",
        "competitor_info": "",
        "final_analysis": ""
    }
    
    result = app.invoke(initial_state)
    
    print("\n\n" + "="*50)
    print("FINAL COMPETITIVE ANALYSIS")
    print("="*50)
    print(result["final_analysis"])
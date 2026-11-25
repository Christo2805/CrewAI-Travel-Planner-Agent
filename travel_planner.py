
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# -----------------------------------
# 1. Load Environment Variables
# -----------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL", "openai/gpt-4o-mini")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Please configure your .env file."
    )

# -----------------------------------
# 2. Get City from User
# -----------------------------------

city = input("Which city would you like to explore? ")

# -----------------------------------
# 3. Configure LLM
# -----------------------------------

llm = LLM(
    model=model_name,
    api_key=api_key
)

# -----------------------------------
# 4. Create Travel Planner Agent
# -----------------------------------

planner = Agent(
    role="Professional Travel Planner",

    goal=(
        "Recommend exactly three interesting and worthwhile "
        "places to visit in the city provided by the user."
    ),

    backstory=(
        "You are an experienced travel planner who specializes "
        "in creating simple and practical travel recommendations. "
        "You help tourists, families, students, and first-time "
        "visitors discover famous attractions, cultural landmarks, "
        "and memorable experiences. You provide clear, accurate, "
        "and useful information without overwhelming travelers "
        "with unnecessary details."
    ),

    llm=llm,
    verbose=True
)

# -----------------------------------
# 5. Create Travel Planning Task
# -----------------------------------

travel_task = Task(

    description=(
        "The user wants to explore {city}. "
        "Identify exactly three of the best places to visit in "
        "{city}. Select interesting attractions that represent "
        "the city's culture, history, architecture, or natural beauty. "
        "For each place, provide its name, a short description, "
        "and one practical visiting tip. Avoid repeating locations."
    ),

    expected_output=(
        "Provide exactly three numbered recommendations. "
        "For each recommendation, include:\n"
        "1. Place name\n"
        "2. A brief description in 1-2 sentences\n"
        "3. One useful visiting tip\n\n"
        "Keep the response concise and easy to understand. "
        "Do not include a lengthy introduction or conclusion. "
        "Do not recommend more or fewer than three places."
    ),

    agent=planner
)

# -----------------------------------
# 6. Create Crew
# -----------------------------------

travel_crew = Crew(
    agents=[planner],
    tasks=[travel_task],
    verbose=True
)

# -----------------------------------
# 7. Execute Crew
# -----------------------------------

print("\n===================================")
print("       CREWAI TRAVEL PLANNER")
print("===================================")

result = travel_crew.kickoff(
    inputs={"city": city}
)

# -----------------------------------
# 8. Display Final Result
# -----------------------------------

print("\n===================================")
print(f" TRAVEL RECOMMENDATIONS: {city.upper()}")
print("===================================\n")

print(result)

print("\n===================================")
print("       PLANNING COMPLETED!")
print("===================================")

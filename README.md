# CITS3011 Intelligent Agent Project

This project is to be completed in groups of two or three students.

This project is marked out of a total of 30 marks and is worth 30% of your unit mark.

A single project mark will be awarded to the group, and every registered member of the group will receive the same mark.

This project is due at 11:59 pm AWST on Friday, 2 October 2026. The unit coordinator reserves the right to extend this deadline in rare but necessary cases.

You are strongly encouraged to submit earlier to avoid any urgent submission issues, and you may receive a late penalty if you are unable to submit by the deadline.


# Description
In this project your group is tasked to research, design, develop, evaluate, and analyse an agent for playing the game Diplomacy.

Your group will be assessed on the performance of the agent it develops and on its written report of the techniques it investigates and develops.

As always, the purpose of this project is to simulate you encountering this as a novel problem. Do not reuse any existing code or computational solutions for Diplomacy.

# Introduction to Diplomacy

Diplomacy is a strategic board game with seven players competing and cooperating to capture supply centres on a map of Europe. This game is very challenging for AI due to the large action space and state space. You need to be creative.

Please read carefully the following information that is necessary for your project.

- Game Rules: https://en.wikibooks.org/wiki/Diplomacy/Rules

The following is the game engine used in this project and its documentation. You will need to read the documentation to find the correct way to retrieve the state information from the engine and interact with the engine: 

- Game Engine: https://github.com/diplomacy/diplomacy
- Documentation: https://diplomacy.readthedocs.io/en/stable/

The following are statistics and databases of the game. You do not need to rely on them to complete the project, but it can help you understand the game:

- Game Statistics: https://vdiplomacy.net/variants.php?variantID=1
- Game Database: https://world-diplomacy-database.com

You can also try playing the game yourself on the online platform below:

- Online Playing: https://webdiplomacy.net/


# Getting Started

Download the attached `the_diplomacy_2026.zip`. Your group should build its agent in `agent_groupnumber.py`, replacing `groupnumber` with your group number.
After renaming the agent file, update the corresponding import in `test.py` and `visualize.py` to use the new filename.
Then you should subclass the Agent class (defined in `agent_baselines.py`) and override the methods and/or implement new methods in your agent.

1. Create a virtual environment using `conda` or `venv` (optional but highly recommended).

2. Install the game engine and other required packages:
```
pip3 install diplomacy tqdm networkx numpy timeout-decorator
```

Or install the packages using the provided `requirements.txt`:
```
pip3 install -r requirements.txt
```

3. Running a test will by default run a large number of games and report the performance of the agent:
```
python3 test.py
```

Your objective is to maximize your expected performance when your agent is dropped into complex (maybe unknown) scenarios.

You can reuse or adapt the testing code `test.py` during the development of your agent. After submission, your agent will be tested under multiple scenarios against multiple baseline agents.


# Game Setup

- Standard Map will be used.
- No Press mode will be used, i.e., no messages among agents.
- Game ends in the year 1920, if there is no winner before 1920.


# Agent Rules

- Your agent must implement the provided Python interface to take part in the game.
- Agents are time-limited and all actions they take must be completed within 1 second.
- Agents are memory-limited and a maximum of 512MB memory can be used.
- Agents are not allowed to save files.
- Agents must not attempt to circumvent or hack the simulation.
- Agents have no connection to the internet and cannot use API calls, e.g., to LLMs.
- Agents have no access to GPUs.

If we believe your agent attempts to violate any of these rules or otherwise undermine the assessment, it may be disqualified and you may receive no mark.


# Report
Your group is required to write a report detailing the techniques it researched and investigated, the reasoning behind its choice of design and technique, and its assessment/analyses of the effectiveness of its agent.

The report should be no more than four A4 pages.

The report should be submitted as a PDF. It must state the group number and the full names and student numbers of all group members. All group members are responsible for the contents of the report.

- If it is not submitted as a PDF, it may receive no mark.
- If it is over length, it may receive no mark, or be truncated and only partly marked.
- If it is illegibly formatted (tiny font, for example) or otherwise unintelligible, it may receive no mark.


# Baseline Agents

Your agent will play the game against baseline agents developed by the teaching staff. There are five different baseline agents.

- **Static Agent**: This is an agent that always takes the default actions, i.e., hold.
- **Random Agent**: This is an agent that always takes random actions.
- **Attitude Agent**: This is an agent that takes random actions, but has attitudes towards other powers, including being friendly, neutral, or hostile. The attitude depends on other players' actions and can change during the game. A friendly agent will never attack you, a hostile one will never support you, and a neutral one can do anything.
- **Greedy Agent**: This is an agent that always takes greedy actions, without long-term planning. Each unit controlled by the agent will move towards and attack the closest supply centre, or support other units if having the same target.
- **Hidden Agent**: This is an unknown agent.


# Scenarios

- **Scenario 1**: Your agent will control a random power. Other powers are all controlled by copies of **Static Agent**.
- **Scenario 2**: Your agent will control a random power. Other powers are controlled by copies of agents randomly chosen from **Random Agent**, **Attitude Agent**, and **Greedy Agent**. The **Random Agent** is less likely to appear than the other two.
- **Scenario 3**: Your agent will control a random power. Other powers are controlled by copies of agents randomly chosen from **Random Agent**, **Attitude Agent**, **Greedy Agent**, and **Hidden Agent**. There will be exactly one **Hidden Agent** in each game. The **Hidden Agent** is a reasonably strong agent with a ~50% win rate in **Scenario 2**.
- **Scenario 4**: All the group agents will be put together to play a multi-round tournament.


# Marking Rubrics

The marking of the agent and the report will be independent of each other. 
The marking of the agent will focus on the performance. 
The marking of the report will focus on the knowledge, thinking, reasoning, and presentation.

**Agent Rubrics (15 pts)**<sup>[1]</sup>:

- **Scenario 1 (5 pts)**
    - The agent achieves >2% win rate, or captures >7 supply centres on average. (1 pts)
    - The agent achieves >20% win rate, or captures >12 supply centres on average. (3 pts)
    - The agent achieves >90% win rate, or captures >16 supply centres on average. (5 pts)
- **Scenario 2 (5 pts)**
    - The agent achieves >2% win rates, or captures >7 supply centres on average. (1 pts)
    - The agent achieves >25% win rates, or captures >10 supply centres on average. (3 pts)
    - The agent achieves >50% win rates, or captures >13 supply centres on average. (5 pts)
- **Scenario 3 (5 pts)**
    - The agent achieves >2% win rate, or captures >7 supply centres on average. (1 pts)
    - The agent achieves >20% win rate, or captures >9 supply centres on average. (3 pts)
    - The agent achieves >40% win rate, or captures >12 supply centres on average. (5 pts)
- **Scenario 4 (Bonus)**<sup>[2]</sup>
    - The agent ranks top 3 among all group agents in **Scenario 4**. (3 bonus pts)

**Report Rubrics (15 pts)**:

- **Basic Technique (6 pts)**
    - Considers and describes one basic technique <sup>[3]</sup> as your basic method. The basic technique can be from those taught in the lectures, or other existing techniques <sup>[4]</sup>. (2 pts)
    - Discusses the motivations of your basic technique and justifies the choice. (2 pts)
    - Evaluates and analyses the effectiveness of your basic technique and reports quantitative experimental results <sup>[5]</sup>. (2 pts)
- **New Techniques (9 pts)**
    - Creates and describes three new techniques <sup>[6]</sup> <sup>[7]</sup> designed by the group for improvements (e.g., improving your basic method). (3 pts, 1 for each)
    - Discusses the motivations of these new techniques and justifies the designs. (3 pts, 1 for each)
    - Evaluates and analyses the effectiveness of these new techniques and reports quantitative experimental results <sup>[5]</sup>. (3 pts, 1 for each)

## Rubric Notes

**[1]**
: The baseline agents have been provided to you. It is allowed to look at the codes of the baseline agents for better understanding. However, if we have reason to believe that your group has plagiarized from these agents (for example, directly copying them to get points in some scenarios), the group may receive no mark.

**[2]**
: The total points after receiving bonus will not exceed 30 points.

**[3]**
: Your group needs to implement the basic technique in the submitted code, and refer to the implementation in the report, even if the group does not use this basic method in its final version of the agent.

**[4]**
: If you use other existing techniques, references must be provided in the report. Reusing any existing code is prohibited.

**[5]**
: Experimental results can be either positive or negative. The marking will focus on whether the results are presented and analysed meaningfully, not the exact numbers.

**[6]**
: The new technique can be, for example, a new heuristic function, a variant of the search process, or a modification of a basic technique. The techniques need to be well-motivated and justified. The three techniques should be distinct; for example, parameter tuning may not be considered as a new technique.

**[7]**
: Your group needs to implement the new techniques in the submitted code, and refer to the implementation in the report. At least three new techniques should be tried and implemented in the code, even if the group does not use all of them in its final version of the agent.


# Submission

Each group must make one submission to LMS. One group member should submit the following three files on behalf of the group:

- `agent_groupnumber.py`: The group's agent. (Maximum 100KB)
- `report_groupnumber.pdf`: The group report as a PDF. (Maximum 4 pages)
- `test_groupnumber.py`: The group's experiments. (Maximum 100KB)

In the filenames above, `groupnumber` should be replaced with the group's assigned group number.


# Allowed Packages:

The agent is allowed to use built-in packages for Python 3 and the following external packages. Use the provided `requirements.txt` to install the consistent version of the following packages:

- diplomacy
- tqdm
- random
- networkx
- numpy
- scipy
- scikit-learn
- timeout-decorator
- simpleai: https://pypi.org/project/simpleai/

The implementation of the textbook is also allowed to use as reference for your coding: https://github.com/aimacode/aima-python.

# LLM/GenAI Policy:

- You are allowed to use LLM for brainstorming, co-designing, and coding assistance.
- You are NOT allowed to use LLM for report writing.
- If any group member uses LLMs, the group must submit one combined extra PDF document recording all the prompts, named `llm_usage_groupnumber.pdf`.
- The agent itself cannot use any LLM.

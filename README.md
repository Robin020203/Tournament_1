Hybrid Annihalators (and Hybrid Hunters)

This code contains 4 different agents:
- HybridSwitch
- HybridDefence
- AggressiveHybridSwitch
- AggressiveHybridDefence

We use the Hybrid Annihalators, consisting of:
- 1 AggressiveHybridSwitch agent
- 1 AggressiveHybridDefence agent

This team consists of two agents that act very similarly. 
They both start offensive, eating pellets and bringing them back in groups until a threshold is reached.
When they're offensive, they're encouraged to eat capsules to make the other team scared, and if so, 
they will both go all in as long as the scared timer is high enough.
After that, the AggressiveHybridDefensive agent stays defensive,
while the AggressiveHybridSwitch agent can switch back to offensive if the other team is defensive

Our previous team was Hybrid Hunters, consisting of:
- 1 HybridSwitch agent
- 1 HybridDefence agent

This team consists of two agents that act very similarly. 
They both start offensive, eating pellets and bringing them back in groups until a threshold is reached.
After that, the HybridDefensive agent stays defensive,
while the HybridSwitch agent can switch back to offensive if the other team is defensive

Only the default libraries delivered with the task were used:

- random
- util
- capture_agents/CaptureAgent
- game/Directions
- util/nearest_point

To start a game against the baseline team, use the following command:

python capture.py -r agents/team_name_1/my_team.py -b baseline_team

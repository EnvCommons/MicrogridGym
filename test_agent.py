"""
Test agent for MicrogridGym environment.

Runs a GPT model against the environment to verify tool interaction.
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import OpenReward


async def main():
    or_client = OpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/MicrogridGym"
    SPLIT = "train"

    environment = or_client.environments.get(name=ENV_NAME)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    for task in tasks[:1]:
        rollout = or_client.rollout.create(
            run_name="microgrid_test",
            rollout_name="test_run",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
        )

        async with environment.session(task=task) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            rollout.log_openai_response(
                message=input_list[0], is_finished=finished
            )

            turn = 0
            max_turns = 30

            while not finished and turn < max_turns:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                rollout.log_openai_response(response.output[-1])
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments)),
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })
                        rollout.log_openai_response(
                            input_list[-1],
                            reward=reward,
                            is_finished=finished,
                        )

                        print(f"Tool: {item.name}")
                        print(f"Reward: {reward:.4f}")

                        if finished:
                            print("FINISHED!")
                            break

                turn += 1

            print(f"Completed in {turn} turns")


if __name__ == "__main__":
    asyncio.run(main())

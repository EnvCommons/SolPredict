"""
test_agent.py - Test agent for solpredict solubility prediction environment

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_agent.py
"""

import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/SolPredict"
    SPLIT = "train"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")
    print(f"Found {len(tools)} tools")

    for task in tasks[:1]:  # Test first task
        print(f"\nStarting task: {task.task_spec}")

        rollout = or_client.rollout.create(
            run_name="solpredict_test",
            rollout_name="test_run",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec
        )

        async with environment.session(task=task, secrets={"openai_api_key": OPENAI_API_KEY}) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            rollout.log_openai_response(message=input_list[0], is_finished=finished)

            turn = 0
            while not finished:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list
                )

                rollout.log_openai_response(response.output[-1])
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        print(f"Tool: {item.name}")
                        print(f"Args: {item.arguments[:200]}..." if len(str(item.arguments)) > 200 else f"Args: {item.arguments}")

                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        # Truncate long outputs for display
                        output_text = tool_result.blocks[0].text
                        if len(output_text) > 500:
                            print(f"Output: {output_text[:500]}...")
                        else:
                            print(f"Output: {output_text}")

                        if reward:
                            print(f"Reward: {reward:.4f}")

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                        })
                        rollout.log_openai_response(
                            input_list[-1],
                            reward=reward,
                            is_finished=finished
                        )

                        if finished:
                            print("\n=== TASK COMPLETED ===")
                            print(f"Final Reward: {reward:.4f}")
                            break

                if turn > 50:  # Safety limit
                    print("Exceeded turn limit")
                    break


if __name__ == "__main__":
    asyncio.run(main())

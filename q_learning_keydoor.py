import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import math
import time

import gymnasium_env


def calculate_state_index(position, key1_status, door1_status, key2_status, door2_status, total_positions):
    state_value = position
    factor = total_positions
    state_value += int(key1_status) * factor
    factor *= 2
    state_value += int(door1_status) * factor
    factor *= 2
    state_value += int(key2_status) * factor
    factor *= 2
    state_value += int(door2_status) * factor
    return int(state_value)


def smooth_data(data, window_size):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def subsample_data(data, max_points=5000):
    if len(data) <= max_points:
        return np.arange(len(data)), data
    step = max(1, len(data) // max_points)
    indices = np.arange(0, len(data), step)
    return indices, np.array(data)[indices]


def execute_learning_algorithm(
        environment_id='gymnasium_env/KeyDoorEnv',
        level_name="map1",
        training_episodes=500000, training_mode=True, show_training=False,
        show_results=True, demo_runs=5, step_limit=None):
    setup_params = {'map_name': level_name, 'interactive_mode': (not training_mode)}
    if not training_mode or show_training:
        setup_params['render_mode'] = "human"
    else:
        setup_params['render_mode'] = None

    config_env = gym.make(environment_id, map_name=level_name, render_mode=None, interactive_mode=False)
    grid_rows, grid_cols = config_env.unwrapped.nrow, config_env.unwrapped.ncol

    if step_limit is None:
        step_limit = grid_rows * grid_cols * 2
        if level_name == "map1":
            step_limit = 500

    config_env.close()

    main_env = gym.make(environment_id, **setup_params)
    main_env = gym.wrappers.TimeLimit(main_env, max_episode_steps=step_limit)

    print(f"Environment setup: Level '{level_name}' ({grid_rows}x{grid_cols})")
    print(f"Training environment ready (step_limit={step_limit})")

    position_count = main_env.observation_space.n
    total_states = position_count * (2 ** 4)
    action_count = main_env.action_space.n
    clean_level_name = level_name.replace('x', '_').replace('-', '_')
    results_folder = f"learning_results_{clean_level_name}"
    os.makedirs(results_folder, exist_ok=True)
    table_file = os.path.join(results_folder, f"value_table.pkl")
    chart_file = os.path.join(results_folder, f'training_progress.png')

    if training_mode:
        value_table = np.zeros((total_states, action_count))
    else:
        if os.path.exists(table_file):
            with open(table_file, "rb") as f:
                value_table = pickle.load(f)
            if value_table.shape != (total_states, action_count):
                value_table = np.zeros((total_states, action_count))
                print("Table size mismatch - created new table")
        else:
            value_table = np.zeros((total_states, action_count))
            print("No existing table found - created new table")

    alpha = 0.05
    gamma = 0.99
    exploration_rate = 1.0
    min_exploration = 0.01
    if training_episodes > 0 and exploration_rate > min_exploration:
        decay_factor = math.pow(min_exploration / exploration_rate, 1.0 / (training_episodes * 0.8))
    else:
        decay_factor = 0.99999

    random_gen = np.random.default_rng()
    reward_history = []
    exploration_history = []

    print(f"Learning Parameters:")
    print(f"Level: {level_name} ({grid_rows}x{grid_cols})")
    print(f"States: {total_states} (Positions: {position_count}, Flags: x16)")
    print(f"Actions: {action_count}, Episodes: {training_episodes}, Max Steps: {step_limit}")
    print(f"Alpha:{alpha}, Gamma:{gamma}, Exploration(start/min):{exploration_rate:.2f}/{min_exploration:.2f}")
    print(f"Table file: {table_file}\n------------------------------------")

    for episode_num in range(training_episodes):
        current_position, game_info = main_env.reset()
        current_state = calculate_state_index(current_position, game_info['has_key1'], game_info['door1_open'],
                                              game_info['has_key2'], game_info['door2_open'], position_count)
        episode_done, time_up = False, False
        episode_score = 0.0
        step_count = 0

        while not episode_done and not time_up:
            step_count += 1
            if training_mode and random_gen.random() < exploration_rate:
                chosen_action = main_env.action_space.sample()
            else:
                chosen_action = np.argmax(value_table[current_state, :])

            new_position, step_reward, episode_done, time_up, new_info = main_env.step(chosen_action)
            next_state = calculate_state_index(new_position, new_info['has_key1'], new_info['door1_open'],
                                               new_info['has_key2'], new_info['door2_open'], position_count)
            episode_score += step_reward

            if training_mode:
                current_value = value_table[current_state, chosen_action]
                max_next_value = np.max(value_table[next_state, :])
                updated_value = current_value + alpha * (step_reward + gamma * max_next_value - current_value)
                value_table[current_state, chosen_action] = updated_value

            current_state = next_state
            if show_training and training_mode:
                main_env.render()

        if training_mode:
            reward_history.append(episode_score)
            exploration_history.append(exploration_rate)
            exploration_rate = max(min_exploration, exploration_rate * decay_factor)

            if (episode_num + 1) % (training_episodes // 200 if training_episodes >= 200 else 1) == 0:
                recent_avg = np.mean(reward_history[-(
                    training_episodes // 100 if training_episodes >= 100 else 1):]) if reward_history else 0
                print(
                    f"Episode {episode_num + 1}/{training_episodes}|Explore:{exploration_rate:.4f}|AvgScore:{recent_avg:.2f}|LastScore:{episode_score:.2f}|Steps:{step_count}")

    if training_mode and value_table is not None:
        print("Saving value table...")
        with open(table_file, 'wb') as f:
            pickle.dump(value_table, f)
        print(f"Value table saved to {table_file}")

        if reward_history:
            fig, ax1 = plt.subplots(figsize=(14, 8))

            max_plot_points = 8000
            smoothing_window = max(1, training_episodes // 150) if training_episodes >= 150 else 1
            smoothing_window = min(smoothing_window, 1000)

            reward_color = 'steelblue'
            exploration_color = 'forestgreen'

            if len(reward_history) > smoothing_window:
                smoothed_rewards = smooth_data(reward_history, smoothing_window)
                smooth_x = np.arange(smoothing_window - 1, len(reward_history))

                if len(smooth_x) > max_plot_points:
                    plot_indices, plot_rewards = subsample_data(smoothed_rewards, max_plot_points)
                    plot_x = smooth_x[plot_indices]
                    ax1.plot(plot_x, plot_rewards, color=reward_color, linewidth=1.5,
                             label=f'Smoothed Score (window={smoothing_window})')
                else:
                    ax1.plot(smooth_x, smoothed_rewards, color=reward_color, linewidth=1.5,
                             label=f'Smoothed Score (window={smoothing_window})')
            else:
                if len(reward_history) > max_plot_points:
                    plot_indices, plot_rewards = subsample_data(reward_history, max_plot_points)
                    ax1.plot(plot_indices, plot_rewards, color=reward_color, alpha=0.7, linewidth=1,
                             label='Raw Score')
                else:
                    ax1.plot(reward_history, color=reward_color, alpha=0.7, linewidth=1, label='Raw Score')

            ax1.set_xlabel('Episode', fontsize=12)
            ax1.set_ylabel('Score', color=reward_color, fontsize=12)
            ax1.tick_params(axis='y', labelcolor=reward_color)
            ax1.grid(True, alpha=0.3, linestyle='--')

            if exploration_history:
                ax2 = ax1.twinx()
                if len(exploration_history) > max_plot_points:
                    exp_indices, exp_values = subsample_data(exploration_history, max_plot_points)
                    ax2.plot(exp_indices, exp_values, color=exploration_color, linestyle='--', linewidth=2,
                             label='Exploration Rate')
                else:
                    ax2.plot(exploration_history, color=exploration_color, linestyle='--', linewidth=2,
                             label='Exploration Rate')

                ax2.set_ylabel('Exploration Rate', color=exploration_color, fontsize=12)
                ax2.tick_params(axis='y', labelcolor=exploration_color)
                ax2.set_ylim(-0.02, 1.02)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
            else:
                ax1.legend(loc='upper right', fontsize=10)

            plt.title(f'Training Progress ({level_name})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"Enhanced progress chart saved to {chart_file}")
            plt.close()

    if show_results and demo_runs > 0 and value_table is not None:
        print(f"\n--- Performance Demo ({demo_runs} episodes) ---")
        demo_params = {'map_name': level_name, 'render_mode': "human", 'interactive_mode': (not training_mode)}
        demo_env = gym.make(environment_id, **demo_params)
        demo_env = gym.wrappers.TimeLimit(demo_env, max_episode_steps=step_limit)
        print(f"Demo environment ({grid_rows}x{grid_cols}) created.")

        if not training_mode and hasattr(demo_env.unwrapped,
                                         'interactive_mode') and demo_env.unwrapped.interactive_mode:
            print("\n--- Interactive Demo Mode ---")
            demo_position, demo_info = demo_env.reset()
            run_score, run_steps = 0.0, 0
            demo_active = True

            while demo_active:
                if not hasattr(demo_env.unwrapped, 'window') or demo_env.unwrapped.window is None:
                    demo_active = False
                    break

                game_finished = getattr(demo_env.unwrapped, 'game_over_pending', False)
                if not game_finished:
                    demo_state = calculate_state_index(demo_position, demo_info['has_key1'], demo_info['door1_open'],
                                                       demo_info['has_key2'], demo_info['door2_open'], position_count)
                    best_action = np.argmax(value_table[demo_state, :])
                    next_position, step_reward, demo_done, demo_timeout, demo_info = demo_env.step(best_action)
                    run_score += step_reward
                    run_steps += 1
                    demo_position = next_position
                    if demo_done or demo_timeout:
                        print(f"Run completed. Score: {run_score:.2f}, Steps: {run_steps}.")

                demo_env.render()
                time.sleep(1 / max(1, getattr(demo_env.unwrapped, 'current_fps', 30)))
        else:
            for demo_idx in range(demo_runs):
                demo_pos, demo_state_info = demo_env.reset()
                demo_state_idx = calculate_state_index(demo_pos, demo_state_info['has_key1'],
                                                       demo_state_info['door1_open'],
                                                       demo_state_info['has_key2'], demo_state_info['door2_open'],
                                                       position_count)
                demo_finished, demo_truncated = False, False
                demo_total_score = 0
                demo_step_count = 0
                print(f"Demo Run {demo_idx + 1}")

                while not demo_finished and not demo_truncated:
                    demo_action = np.argmax(value_table[demo_state_idx, :])
                    next_demo_pos, demo_reward, demo_finished, demo_truncated, next_demo_info = demo_env.step(
                        demo_action)
                    demo_state_idx = calculate_state_index(next_demo_pos, next_demo_info['has_key1'],
                                                           next_demo_info['door1_open'], next_demo_info['has_key2'],
                                                           next_demo_info['door2_open'], position_count)
                    demo_total_score += demo_reward
                    demo_step_count += 1
                    if demo_env.render_mode == "human":
                        demo_env.render()

                print(f"Demo Run {demo_idx + 1} completed. Score: {demo_total_score:.2f}, Steps: {demo_step_count}")

        demo_env.close()

    main_env.close()
    print("Algorithm execution completed.")


if __name__ == '__main__':
    selected_level = "map1"

    if selected_level == "map1":
        episode_count = 75000
        max_steps = 500

    training_enabled = True
    show_during_training = False
    show_final_demo = True
    demo_episode_count = 2

    print(f"--- Starting Learning Algorithm (KeyDoor), Level: '{selected_level}' ---")
    execute_learning_algorithm(level_name=selected_level, training_episodes=episode_count,
                               training_mode=training_enabled,
                               show_training=show_during_training, show_results=show_final_demo,
                               demo_runs=demo_episode_count, step_limit=max_steps)

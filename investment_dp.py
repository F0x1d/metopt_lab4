import numpy as np
from itertools import product
from typing import Dict, Tuple, List
import pandas as pd
import sys

# Начальные объёмы активов (д.е.)
INITIAL_CB1 = 100  # ЦБ1
INITIAL_CB2 = 800  # ЦБ2
INITIAL_DEP = 400  # Депозиты
INITIAL_CASH = 600  # Свободные средства

# Шаг управления = 25% от начального объёма
STEP_CB1 = INITIAL_CB1 // 4  # 25 д.е.
STEP_CB2 = INITIAL_CB2 // 4  # 200 д.е.
STEP_DEP = INITIAL_DEP // 4  # 100 д.е.

# Комиссии брокеров (при покупке)
COMMISSION_CB1 = 0.04  # 4%
COMMISSION_CB2 = 0.07  # 7%
COMMISSION_DEP = 0.05  # 5%

# Минимальные ограничения на объёмы
MIN_CB1 = 30
MIN_CB2 = 150
MIN_DEP = 100

# Количество этапов
NUM_STAGES = 3

# Вероятности событий на каждом этапе [благопр., нейтр., негатив.]
PROBABILITIES = {
    1: [0.60, 0.30, 0.10],  # Этап 1
    2: [0.30, 0.20, 0.50],  # Этап 2
    3: [0.40, 0.40, 0.20],  # Этап 3
}

# Коэффициенты изменения стоимости на каждом этапе
# [благопр., нейтр., негатив.]
RETURNS_CB1 = {
    1: [1.20, 1.05, 0.80],
    2: [1.40, 1.05, 0.60],
    3: [1.15, 1.05, 0.70],
}

RETURNS_CB2 = {
    1: [1.10, 1.02, 0.95],
    2: [1.15, 1.00, 0.90],
    3: [1.12, 1.01, 0.94],
}

RETURNS_DEP = {
    1: [1.07, 1.03, 1.00],
    2: [1.01, 1.00, 1.00],
    3: [1.05, 1.01, 1.00],
}


def print_progress_bar(
    iteration, total, prefix="", suffix="", length=40, fill="█", empty="░"
):
    """
    Отображает прогресс-бар в консоли.
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()
    if iteration == total:
        print()


class InvestmentDP:
    """
    Класс для решения задачи оптимального управления инвестиционным портфелем
    методом динамического программирования.
    """

    def __init__(self):
        # Функция Беллмана: V[stage][state] = (max_expected_value, optimal_action)
        self.V: Dict[int, Dict[Tuple, Tuple[float, Tuple]]] = {}

        # Достижимые состояния для каждого этапа (генерируются вперёд)
        self.reachable_states: Dict[int, set] = {}

        # Оптимальные решения для прямого прохода
        self.optimal_decisions = {}

        # Статистика выполнения
        self.stats = {"states_processed": 0, "actions_evaluated": 0}

    def get_possible_states(self, stage: int) -> List[Tuple[int, int, int, int]]:
        """
        Генерирует все возможные состояния для данного этапа.
        Состояние: (cb1, cb2, dep, cash) - объёмы активов и свободные средства.
        """
        states = []

        # Диапазоны возможных значений с учётом шагов управления
        # Для упрощения генерируем состояния кратные шагам
        for cb1 in range(0, INITIAL_CB1 + INITIAL_CASH + 200, STEP_CB1):
            for cb2 in range(0, INITIAL_CB2 + INITIAL_CASH + 400, STEP_CB2):
                for dep in range(0, INITIAL_DEP + INITIAL_CASH + 200, STEP_DEP):
                    # Вычисляем свободные средства
                    total_initial = (
                        INITIAL_CB1 + INITIAL_CB2 + INITIAL_DEP + INITIAL_CASH
                    )
                    cash = total_initial - cb1 - cb2 - dep

                    # Проверяем ограничения
                    if cash >= 0 and cb1 >= 0 and cb2 >= 0 and dep >= 0:
                        states.append((cb1, cb2, dep, cash))

        return states

    def get_possible_actions(
        self, state: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """
        Генерирует все возможные управляющие воздействия для данного состояния.
        Действие: (delta_cb1, delta_cb2, delta_dep) - изменения объёмов.
        Положительные значения - покупка, отрицательные - продажа.
        """
        cb1, cb2, dep, cash = state
        actions = []

        # Возможные изменения для каждого актива: -2, -1, 0, +1, +2 шага
        for d_cb1 in range(-4, 5):
            for d_cb2 in range(-4, 5):
                for d_dep in range(-4, 5):
                    delta_cb1 = d_cb1 * STEP_CB1
                    delta_cb2 = d_cb2 * STEP_CB2
                    delta_dep = d_dep * STEP_DEP

                    # Новые объёмы активов
                    new_cb1 = cb1 + delta_cb1
                    new_cb2 = cb2 + delta_cb2
                    new_dep = dep + delta_dep

                    # Проверяем минимальные ограничения
                    if new_cb1 < MIN_CB1 or new_cb2 < MIN_CB2 or new_dep < MIN_DEP:
                        continue

                    # Расчёт изменения денежных средств
                    # При покупке платим комиссию
                    cost = 0
                    if delta_cb1 > 0:
                        cost += delta_cb1 * (1 + COMMISSION_CB1)
                    else:
                        cost += delta_cb1  # При продаже получаем полную стоимость

                    if delta_cb2 > 0:
                        cost += delta_cb2 * (1 + COMMISSION_CB2)
                    else:
                        cost += delta_cb2

                    if delta_dep > 0:
                        cost += delta_dep * (1 + COMMISSION_DEP)
                    else:
                        cost += delta_dep

                    new_cash = cash - cost

                    # Проверяем достаточность средств
                    if new_cash >= 0:
                        actions.append((delta_cb1, delta_cb2, delta_dep))

        return actions

    def apply_action(
        self, state: Tuple[int, int, int, int], action: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Применяет управляющее воздействие к состоянию и возвращает новое состояние.
        """
        cb1, cb2, dep, cash = state
        delta_cb1, delta_cb2, delta_dep = action

        new_cb1 = cb1 + delta_cb1
        new_cb2 = cb2 + delta_cb2
        new_dep = dep + delta_dep

        # Расчёт затрат с учётом комиссий
        cost = 0
        if delta_cb1 > 0:
            cost += delta_cb1 * (1 + COMMISSION_CB1)
        else:
            cost += delta_cb1

        if delta_cb2 > 0:
            cost += delta_cb2 * (1 + COMMISSION_CB2)
        else:
            cost += delta_cb2

        if delta_dep > 0:
            cost += delta_dep * (1 + COMMISSION_DEP)
        else:
            cost += delta_dep

        new_cash = cash - cost

        return (new_cb1, new_cb2, new_dep, new_cash)

    def calculate_expected_value_after_stage(
        self, state: Tuple[int, int, int, int], stage: int
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Вычисляет возможные состояния после случайного события на этапе
        и соответствующие вероятности.

        Возвращает список кортежей: (новое_состояние, вероятность)
        """
        cb1, cb2, dep, cash = state
        probs = PROBABILITIES[stage]
        returns_cb1 = RETURNS_CB1[stage]
        returns_cb2 = RETURNS_CB2[stage]
        returns_dep = RETURNS_DEP[stage]

        outcomes = []
        for i, prob in enumerate(probs):
            new_cb1 = cb1 * returns_cb1[i]
            new_cb2 = cb2 * returns_cb2[i]
            new_dep = dep * returns_dep[i]
            new_cash = cash  # Денежные средства не меняются

            outcomes.append(((new_cb1, new_cb2, new_dep, new_cash), prob))

        return outcomes

    def terminal_value(self, state: Tuple[float, float, float, float]) -> float:
        """
        Терминальная функция ценности - общая стоимость портфеля.
        """
        cb1, cb2, dep, cash = state
        return cb1 + cb2 + dep + cash

    def backward_pass(self):
        """
        Обратный проход алгоритма динамического программирования.
        Вычисляет функцию Беллмана от последнего этапа к первому.
        """
        for stage in range(NUM_STAGES, 0, -1):
            self.V[stage] = {}

            states = self.generate_reachable_states(stage)
            total_states = len(states)

            states_processed = 0
            actions_evaluated = 0

            for i, state in enumerate(states):
                best_value = -float("inf")
                best_action = None

                actions = self.get_possible_actions(state)

                for action in actions:
                    actions_evaluated += 1

                    # Применяем действие
                    state_after_action = self.apply_action(state, action)

                    # Вычисляем мат. ожидание ценности после случайного события
                    expected_value = 0
                    outcomes = self.calculate_expected_value_after_stage(
                        state_after_action, stage
                    )

                    for outcome_state, prob in outcomes:
                        if stage == NUM_STAGES:
                            # Терминальный этап
                            expected_value += prob * self.terminal_value(outcome_state)
                        else:
                            # Ищем ценность следующего этапа
                            # Округляем для поиска в словаре
                            rounded_state = self.round_state_precise(outcome_state)
                            if rounded_state in self.V[stage + 1]:
                                expected_value += (
                                    prob * self.V[stage + 1][rounded_state][0]
                                )
                            else:
                                # Это не должно происходить при правильной генерации состояний
                                raise ValueError(
                                    f"Состояние {rounded_state} не найдено в V[{stage + 1}]. "
                                    f"Ошибка в генерации достижимых состояний."
                                )

                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = action

                self.V[stage][state] = (best_value, best_action)
                states_processed += 1

                # Обновляем прогресс-бар каждые 100 состояний или в конце
                if states_processed % 100 == 0 or states_processed == total_states:
                    print_progress_bar(
                        states_processed,
                        total_states,
                        prefix=f"Обработка состояний",
                        suffix=f"({states_processed}/{total_states}, действий: {actions_evaluated})",
                        length=30,
                    )

            # Обновляем общую статистику
            self.stats["states_processed"] += states_processed
            self.stats["actions_evaluated"] += actions_evaluated

    def generate_all_reachable_states_forward(self):
        """
        Генерирует все достижимые состояния ВПЕРЁД от начального состояния.
        Это гарантирует, что каждое состояние в backward pass будет найдено.
        """
        # Этап 1: только начальное состояние
        initial_state = (INITIAL_CB1, INITIAL_CB2, INITIAL_DEP, INITIAL_CASH)
        self.reachable_states[1] = {initial_state}

        # Этапы 2 и 3: генерируем из предыдущих состояний
        for stage in range(1, NUM_STAGES):
            next_stage = stage + 1
            self.reachable_states[next_stage] = set()

            for state in self.reachable_states[stage]:
                # Все возможные действия из этого состояния
                actions = self.get_possible_actions(state)

                for action in actions:
                    # Применяем действие
                    state_after_action = self.apply_action(state, action)

                    # Все возможные исходы случайного события
                    outcomes = self.calculate_expected_value_after_stage(
                        state_after_action, stage
                    )

                    for outcome_state, prob in outcomes:
                        # Округляем до копеек (2 знака после запятой)
                        rounded = self.round_state_precise(outcome_state)
                        self.reachable_states[next_stage].add(rounded)

    def round_state_precise(
        self, state: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Округляет состояние до копеек (2 знака после запятой).
        """
        cb1, cb2, dep, cash = state
        return (round(cb1, 2), round(cb2, 2), round(dep, 2), round(cash, 2))

    def generate_reachable_states(self, stage: int) -> List[Tuple]:
        """
        Возвращает предварительно сгенерированные достижимые состояния для этапа.
        """
        return list(self.reachable_states[stage])

    def round_state(
        self, state: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        """
        Округляет состояние до ближайших кратных шагам значений.
        """
        cb1, cb2, dep, cash = state

        rounded_cb1 = round(cb1 / STEP_CB1) * STEP_CB1
        rounded_cb2 = round(cb2 / STEP_CB2) * STEP_CB2
        rounded_dep = round(dep / STEP_DEP) * STEP_DEP
        rounded_cash = round(cash / 50) * 50

        return (int(rounded_cb1), int(rounded_cb2), int(rounded_dep), int(rounded_cash))

    def forward_pass(self):
        """
        Прямой проход - определение оптимальной стратегии от начального состояния.
        Показывает оптимальные решения и симулирует конкретные сценарии.
        """
        initial_state = (INITIAL_CB1, INITIAL_CB2, INITIAL_DEP, INITIAL_CASH)

        # Получаем оптимальное решение для начального состояния
        if initial_state in self.V[1]:
            expected_value, optimal_action = self.V[1][initial_state]
        else:
            optimal_action = self.find_best_action_for_state(initial_state, 1)

        decisions = [optimal_action]

        # Применяем решение
        state_after_action = self.apply_action(initial_state, optimal_action)

        # Показываем возможные исходы после этапа 1
        outcomes_stage1 = self.calculate_expected_value_after_stage(
            state_after_action, 1
        )

        # Симуляция конкретных сценариев
        print("\n" + "=" * 70)
        print("СИМУЛЯЦИЯ КОНКРЕТНЫХ СЦЕНАРИЕВ")
        print("=" * 70)

        scenario_sequences = [
            ([0, 0, 0], "Все благоприятные"),
            ([2, 2, 2], "Все негативные"),
            ([1, 1, 1], "Все нейтральные"),
            ([0, 2, 0], "Благоприятный-Негативный-Благоприятный"),
            ([2, 0, 1], "Негативный-Благоприятный-Нейтральный"),
        ]

        for scenario_indices, scenario_name in scenario_sequences:
            prob = self._calculate_scenario_probability(scenario_indices)
            final_value, trajectory, scenario_decisions = self._simulate_scenario(
                initial_state, scenario_indices, verbose=False
            )
            print(f"\n{scenario_name} (вероятность {prob:.2%}):")
            print(f"  Итоговая стоимость: {final_value:.2f} д.е.")
            print(f"  Доходность: {(final_value / sum(initial_state) - 1) * 100:.2f}%")

        self.optimal_decisions = decisions
        return [], decisions

    def _print_state(self, state: Tuple) -> None:
        """Вспомогательный метод для печати состояния."""
        print(f"  ЦБ1: {state[0]:.2f} д.е.")
        print(f"  ЦБ2: {state[1]:.2f} д.е.")
        print(f"  Депозиты: {state[2]:.2f} д.е.")
        print(f"  Свободные средства: {state[3]:.2f} д.е.")
        print(f"  Общая стоимость: {sum(state):.2f} д.е.")

    def _print_action(self, action: Tuple[int, int, int]) -> None:
        """Вспомогательный метод для печати действия."""
        print(f"\nОптимальное решение:")
        labels = ["ЦБ1", "ЦБ2", "Депозиты"]
        ops_pos = ["покупка", "покупка", "пополнение"]
        ops_neg = ["продажа", "продажа", "снятие"]

        for i, (label, delta) in enumerate(zip(labels, action)):
            print(f"  Δ {label}: {delta:+d} д.е.", end="")
            if delta > 0:
                print(f" ({ops_pos[i]})")
            elif delta < 0:
                print(f" ({ops_neg[i]})")
            else:
                print(" (без изменений)")

    def _print_action_summary(self, action: Tuple[int, int, int]) -> None:
        """Вспомогательный метод для краткой печати действия."""
        action_desc = []
        if action[0] != 0:
            op = "Купить" if action[0] > 0 else "Продать"
            action_desc.append(f"  {op} ЦБ1 на {abs(action[0])} д.е.")
        if action[1] != 0:
            op = "Купить" if action[1] > 0 else "Продать"
            action_desc.append(f"  {op} ЦБ2 на {abs(action[1])} д.е.")
        if action[2] != 0:
            op = "Пополнить" if action[2] > 0 else "Снять с"
            action_desc.append(
                f"  {op} депозит{'ы' if action[2] > 0 else 'ов'} на {abs(action[2])} д.е."
            )

        if action_desc:
            for desc in action_desc:
                print(desc)
        else:
            print("  Без изменений")

    def _print_outcomes(self, outcomes: List[Tuple]) -> None:
        """Вспомогательный метод для печати возможных исходов."""
        scenario_names = ["Благоприятный", "Нейтральный", "Негативный"]
        for i, (outcome_state, prob) in enumerate(outcomes):
            print(f"\n  {scenario_names[i]} (вероятность {prob:.0%}):")
            print(f"    ЦБ1: {outcome_state[0]:.2f} д.е.")
            print(f"    ЦБ2: {outcome_state[1]:.2f} д.е.")
            print(f"    Депозиты: {outcome_state[2]:.2f} д.е.")
            print(f"    Свободные средства: {outcome_state[3]:.2f} д.е.")
            print(f"    Общая стоимость: {sum(outcome_state):.2f} д.е.")

    def _simulate_scenario(
        self, initial_state: Tuple, scenario_indices: List[int], verbose: bool = False
    ) -> Tuple[float, List, List]:
        """
        Симулирует конкретный сценарий с заданной последовательностью событий.

        scenario_indices: список индексов событий [0=благоприятный, 1=нейтральный, 2=негативный]
        verbose: если True, выводит подробную информацию о каждом этапе

        Возвращает: (итоговая стоимость, траектория, список решений)
        """
        current_state = initial_state
        trajectory = [current_state]
        decisions = []

        for stage in range(1, NUM_STAGES + 1):
            # Находим оптимальное действие для текущего состояния
            rounded_state = self.round_state_precise(current_state)

            if rounded_state in self.V[stage]:
                _, optimal_action = self.V[stage][rounded_state]
            else:
                optimal_action = self.find_best_action_for_state(current_state, stage)

            decisions.append(optimal_action)

            # Применяем действие
            state_after_action = self.apply_action(current_state, optimal_action)

            # Применяем случайное событие (конкретный исход)
            event_idx = scenario_indices[stage - 1]
            cb1 = state_after_action[0] * RETURNS_CB1[stage][event_idx]
            cb2 = state_after_action[1] * RETURNS_CB2[stage][event_idx]
            dep = state_after_action[2] * RETURNS_DEP[stage][event_idx]
            cash = state_after_action[3]

            current_state = (cb1, cb2, dep, cash)
            trajectory.append(current_state)

        final_value = sum(current_state)
        return final_value, trajectory, decisions

    def _calculate_scenario_probability(self, scenario_indices: List[int]) -> float:
        """Вычисляет вероятность конкретного сценария."""
        prob = 1.0
        for stage in range(1, NUM_STAGES + 1):
            event_idx = scenario_indices[stage - 1]
            prob *= PROBABILITIES[stage][event_idx]
        return prob

    def find_best_action_for_state(
        self, state: Tuple, stage: int
    ) -> Tuple[int, int, int]:
        """
        Находит лучшее действие для состояния методом перебора.
        """
        best_value = -float("inf")
        best_action = (0, 0, 0)

        actions = self.get_possible_actions(state)

        for action in actions:
            state_after_action = self.apply_action(state, action)
            expected_value = 0
            outcomes = self.calculate_expected_value_after_stage(
                state_after_action, stage
            )

            for outcome_state, prob in outcomes:
                if stage == NUM_STAGES:
                    expected_value += prob * self.terminal_value(outcome_state)
                else:
                    rounded_state = self.round_state_precise(outcome_state)
                    if rounded_state in self.V[stage + 1]:
                        expected_value += prob * self.V[stage + 1][rounded_state][0]
                    else:
                        # Для симуляции сценариев используем терминальную оценку
                        # (это может происходить только при симуляции, не в backward pass)
                        expected_value += prob * self.terminal_value(outcome_state)

            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        return best_action

    def solve(self):
        """
        Основной метод решения задачи.
        """
        # Генерируем достижимые состояния вперёд
        self.generate_all_reachable_states_forward()

        # Выполняем обратный проход
        self.backward_pass()

        # Выполняем прямой проход
        trajectory, decisions = self.forward_pass()

        # Итоговый результат
        print("\n" + "=" * 70)
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
        print("=" * 70)

        initial_state = (INITIAL_CB1, INITIAL_CB2, INITIAL_DEP, INITIAL_CASH)
        initial_value = sum(initial_state)

        if initial_state in self.V[1]:
            optimal_expected_value = self.V[1][initial_state][0]
            expected_profit = optimal_expected_value - initial_value
            expected_return = (optimal_expected_value / initial_value - 1) * 100

            print(f"\nНачальная стоимость портфеля: {initial_value:.2f} д.е.")
            print(f"Ожидаемая стоимость портфеля: {optimal_expected_value:.2f} д.е.")
            print(f"Ожидаемая прибыль: {expected_profit:.2f} д.е.")
            print(f"Ожидаемая доходность: {expected_return:.2f}%")

        # Оптимальное решение на этапе 1
        if decisions:
            action = decisions[0]
            print(f"\nОптимальное решение на этапе 1:")
            self._print_action_summary(action)

        return trajectory, decisions


def main():
    """
    Главная функция программы.
    """
    solver = InvestmentDP()
    trajectory, decisions = solver.solve()


if __name__ == "__main__":
    main()

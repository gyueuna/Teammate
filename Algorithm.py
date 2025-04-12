import random
import numpy as np
from scipy.optimize import linear_sum_assignment

# 최소/최대 팀 인원수 설정
MIN_TEAM_SIZE = 3
MAX_TEAM_SIZE = 4

# 이상적인 팀 크기 (조건을 엄격히 지키므로 사용하지 않음)
IDEAL_TEAM_SIZE = (MIN_TEAM_SIZE + MAX_TEAM_SIZE) / 2.0

# --- 1. 학생 데이터 생성 ---
def generate_students(n):
    """
    n명의 학생 데이터를 생성.
    각 학생은:
      - 'skill': 0 ~ 10 사이의 실력 점수,
      - 'schedule': 평일(mon~fri)에서 오전 8시부터 24시까지의 시간 슬롯 중 무작위로 3~6개 선택한 (day, hour) 튜플 집합,
      - 'interests': 미리 정의된 관심사 리스트에서 무작위로 1~3개 선택한 집합.
    """
    students = []
    possible_interests = ["AI", "ML", "Data", "Web", "Mobile", "Security"]
    # 가능한 시간 슬롯: 평일(mon~fri), 시간은 8시부터 24시까지 (range(8, 25))
    days = ["mon", "tue", "wed", "thu", "fri", "Sat", "Sun"]
    possible_time_slots = [(d, h) for d in days for h in range(8, 25)]

    for i in range(n):
        skill = random.uniform(0, 10)
        time_slots = set(random.sample(possible_time_slots, random.randint(3, 6)))
        interests = set(random.sample(possible_interests, random.randint(1, 3)))
        students.append({
            "id": i,
            "skill": skill,
            "schedule": time_slots,  # (day, hour) 튜플들의 set
            "interests": interests
        })
    return students

# --- 2. 전처리: 학생 데이터를 벡터화 (실력, 관심사) ---
def preprocess_students(students):
    """
    학생의 실력과 관심사는 numpy 배열로, 시간표는 원래 set 형태 그대로 리스트로 반환.
    """
    n = len(students)
    skills = np.array([s["skill"] for s in students])
    # schedules: 각 학생이 가진 schedule (set)들의 리스트
    schedules = [s["schedule"] for s in students]
    # 관심사: 미리 정의된 6개 관심사에 대해 0/1 배열
    possible_interests = ["AI", "ML", "Data", "Web", "Mobile", "Security"]
    interests = np.zeros((n, len(possible_interests)), dtype=int)
    for i, s in enumerate(students):
        for interest in s["interests"]:
            idx = possible_interests.index(interest)
            interests[i, idx] = 1
    return skills, schedules, interests

# --- 3. 유효한 분할(Partition) 생성 ---
def partition_students(n):
    """
    n명을 3명 또는 4명으로 분할하여 합이 n이 되도록 하는 팀 크기 리스트를 반환.
    가능한 해 중 팀 수(x+y)를 최대화하는 분할을 선택.
    """
    best_total = -1
    best_sizes = None
    # x 팀이 3명, y 팀이 4명: 3*x + 4*y = n
    for x in range(n // MIN_TEAM_SIZE + 1):
        remainder = n - MIN_TEAM_SIZE * x
        if remainder < 0:
            continue
        if remainder % MAX_TEAM_SIZE == 0:
            y = remainder // MAX_TEAM_SIZE
            total = x + y
            if total > best_total:
                best_total = total
                best_sizes = [MIN_TEAM_SIZE]*x + [MAX_TEAM_SIZE]*y
    if best_sizes is None:
        best_sizes = [MAX_TEAM_SIZE] * (n // MAX_TEAM_SIZE)
    return best_sizes

def create_valid_partition(n):
    """
    n명의 학생을 partition_students()를 이용해 무작위로 배분하여,
    학생별 팀 번호를 담은 numpy 배열(solution)을 생성.
    이 함수는 항상 팀원 수가 MIN_TEAM_SIZE와 MAX_TEAM_SIZE 조건을 만족하는 해를 생성합니다.
    """
    sizes = partition_students(n)
    random.shuffle(sizes)  # 팀 순서를 랜덤하게 섞음
    assignment = np.empty(n, dtype=int)
    indices = np.arange(n)
    np.random.shuffle(indices)
    team_id = 0
    start = 0
    for size in sizes:
        assignment[indices[start:start+size]] = team_id
        team_id += 1
        start += size
    return assignment

# --- 4. 해 디코딩: 모든 팀 반환 (여기서는 팀원 수가 어떻게 되어 있든 모두 반환)
def decode_solution(solution):
    teams = {}
    for team in np.unique(solution):
        indices = np.where(solution == team)[0]
        teams[team] = indices
    return teams

# --- 5. Fitness 함수 ---
def fitness(solution, skills, schedules, interests):
    """
    주어진 해(solution)에 대해 모든 팀을 평가합니다.
    평가 요소:
      - 실력: 팀원 평균 실력의 표준편차가 낮을수록 좋음.
      - 시간표: 팀원들의 schedule set들의 교집합/합집합 비율.
      - 관심사: 팀 내 학생 쌍의 Jaccard 유사도.
    또한, 각 팀의 팀원 수가 이상적 범위(3~4명)와 얼마나 차이나는지 패널티를 부여합니다.
    최종 score = (0.4*실력 + 0.3*시간표 + 0.3*관심사) * (1 - avg_penalty),
    여기서 avg_penalty = (sum_over_all_students |team_size - ideal| / ideal) / total_students.
    """
    teams = decode_solution(solution)
    total_students = len(solution)

    team_avg_skills = []
    time_scores = []
    interest_scores = []
    total_penalty = 0.0
    for indices in teams.values():
        team_size = len(indices)
        # penalty: 팀원 수가 이상적(3.5명)에서 벗어난 정도
        penalty = abs(team_size - ((MIN_TEAM_SIZE+MAX_TEAM_SIZE)/2)) / ((MIN_TEAM_SIZE+MAX_TEAM_SIZE)/2)
        total_penalty += penalty * team_size

        avg_skill = np.mean(skills[indices])
        team_avg_skills.append(avg_skill)

        # 시간표: schedule set들의 교집합 vs 합집합
        schedule_sets = [schedules[i] for i in indices]
        common = set.intersection(*schedule_sets)
        union = set.union(*schedule_sets)
        time_ratio = len(common) / len(union) if len(union) > 0 else 0
        time_scores.append(time_ratio)

        # 관심사: 팀 내 모든 학생 쌍 Jaccard 유사도
        team_size_local = len(indices)
        current_triu = np.triu_indices(team_size_local, k=1)
        team_int = interests[indices]
        inter_mat = team_int.dot(team_int.T)
        team_sum = np.sum(team_int, axis=1)
        union_mat = team_sum.reshape(-1, 1) + team_sum.reshape(1, -1) - inter_mat
        jaccard = inter_mat[current_triu] / (union_mat[current_triu] + 1e-6)
        interest_scores.append(np.mean(jaccard) if jaccard.size > 0 else 0)

    skill_std = np.std(team_avg_skills)
    skill_score = 1 / (1 + skill_std)
    time_score = np.mean(time_scores)
    interest_score = np.mean(interest_scores)

    base_score = 0.4 * skill_score + 0.3 * time_score + 0.3 * interest_score
    avg_penalty = total_penalty / total_students  # 평균 페널티 per student
    final_score = base_score * (1 - avg_penalty)
    return max(final_score, 0)

# --- 6. Repair 함수: 해(solution)이 반드시 팀원 수 조건을 만족하도록 보정
def repair_solution(solution, n):
    """
    주어진 solution에서 각 팀의 학생 수가 MIN_TEAM_SIZE ~ MAX_TEAM_SIZE를 만족하는지 확인.
    만약 조건에 맞지 않으면, 전체 해를 create_valid_partition(n)으로 대체하여 반환.
    (더 정교한 repair 알고리즘을 구현할 수도 있으나, 여기서는 간단하게 처리합니다.)
    """
    teams = decode_solution(solution)
    valid = True
    for indices in teams.values():
        if not (MIN_TEAM_SIZE <= len(indices) <= MAX_TEAM_SIZE):
            valid = False
            break
    if valid and sum(len(indices) for indices in teams.values()) == n:
        return solution
    else:
        return create_valid_partition(n)

# --- 7. 초기 집단 생성 ---
def create_initial_population(pop_size, n):
    population = []
    for _ in range(pop_size):
        population.append(create_valid_partition(n))
    return population

# --- 8. 토너먼트 선택 ---
def tournament_selection(population, skills, schedules, interests, tournament_size=3):
    selected = random.sample(population, tournament_size)
    selected = sorted(selected, key=lambda ind: fitness(ind, skills, schedules, interests), reverse=True)
    return selected[0]

# --- 9. 단일 지점 교차 ---
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# --- 10. 돌연변이 ---
def mutate(individual, mutation_rate=0.1, n=None):
    if n is None:
        n = len(individual)
    team_ids = list(np.unique(individual))
    for i in range(n):
        if random.random() < mutation_rate:
            individual[i] = random.choice(team_ids)
    return individual

# --- 11. GA 메인 함수 ---
def genetic_algorithm(n, skills, schedules, interests, pop_size=30, generations=50):
    population = create_initial_population(pop_size, n)
    best_solution = None
    best_fit = -1.0
    for gen in range(generations):
        new_population = []
        population = sorted(population, key=lambda ind: fitness(ind, skills, schedules, interests), reverse=True)
        current_best = population[0].copy()
        current_fit = fitness(current_best, skills, schedules, interests)
        if current_fit > best_fit:
            best_fit = current_fit
            best_solution = current_best.copy()
        new_population.append(current_best.copy())
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, skills, schedules, interests)
            parent2 = tournament_selection(population, skills, schedules, interests)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate=0.1, n=n)
            child2 = mutate(child2, mutation_rate=0.1, n=n)
            # repair 단계를 통해 반드시 팀원 수 조건을 지키도록 함
            child1 = repair_solution(child1, n)
            child2 = repair_solution(child2, n)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]
        if gen % 5 == 0:
            print(f"Generation {gen}, Best fitness: {best_fit:.4f}")
    return best_solution

# --- 메인 실행부 ---
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    n = 100  # 학생 수 100명dd
    students = generate_students(n)
    skills, schedules, interests = preprocess_students(students)

    best_solution = genetic_algorithm(n, skills, schedules, interests, pop_size=30, generations=50)
    best_fit_value = fitness(best_solution, skills, schedules, interests)
    print("\n최종 Best fitness score:", best_fit_value)

    teams = decode_solution(best_solution)
    print("\n최종 팀 구성 결과:")
    for team_id, indices in teams.items():
        print(f"\nTeam {team_id} (Size: {len(indices)}):")
        for idx in indices:
            s = students[idx]
            print(f"  Student {s['id']}, Skill: {s['skill']:.2f}, Schedule: {s['schedule']}, Interests: {s['interests']}")


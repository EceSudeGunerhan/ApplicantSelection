import numpy as np
import pandas as pd

def generate_hiring_data(n_samples=10000, label_noise_ratio=0.05, random_state=42):
    np.random.seed(random_state)

    exp_years = np.random.randint(0, 10, n_samples)
    tech_score = np.random.uniform(0, 100, n_samples)
    exp_noise = np.random.normal(0, 1, n_samples)
    score_noise = np.random.normal(0, 5, n_samples)

    experience_years = exp_years + exp_noise
    technical_score = tech_score + score_noise

    hire_label = []
    for year, score in zip(experience_years, technical_score):
        if year >= 2 and score >= 60:
            hire_label.append(1)
        else:
            hire_label.append(0)
    hire_label = np.array(hire_label)

    noise_indices = np.random.choice(n_samples, size=int(label_noise_ratio * n_samples), replace=False)
    hire_label[noise_indices] = 1 - hire_label[noise_indices]

    df = pd.DataFrame({
        "experience_years": experience_years,
        "technical_score": technical_score,
        "hire_label":hire_label
    })

    return df

if __name__ == "__main__":
    print(generate_hiring_data())
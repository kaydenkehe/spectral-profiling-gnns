import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_physics_questions():
    """Plot SLP profiles for Physics and Questions datasets side by side."""
    
    metrics_path = Path(__file__).parent / "metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    physics_data = metrics["Physics"]
    questions_data = metrics["Questions"]
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharex=False, sharey=False)
    
    physics_eigenvalues = np.array(physics_data["eigenvalues"])
    physics_cdf = np.array(physics_data["cdf"])
    axes[0].step(physics_eigenvalues, physics_cdf, where='post', linewidth=1.5)
    axes[0].set_title(f'Physics (h={physics_data["homophily"]:.2f})', fontsize=10)
    axes[0].set_xlim(0, 2)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.3)
    
    questions_eigenvalues = np.array(questions_data["eigenvalues"])
    questions_cdf = np.array(questions_data["cdf"])
    axes[1].step(questions_eigenvalues, questions_cdf, where='post', linewidth=1.5)
    axes[1].set_title(f'Questions (h={questions_data["homophily"]:.2f})', fontsize=10)
    axes[1].set_xlim(0, 2)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    
    for ax in axes:
        ax.tick_params(labelbottom=False, labelleft=False)
    
    for ax in axes:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(r'$\lambda^*$')
    
    axes[0].tick_params(labelleft=True)
    axes[0].set_ylabel(r'$\Pi(\lambda^*)$')
    
    axes[1].legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "physics_questions_slp.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_physics_questions()

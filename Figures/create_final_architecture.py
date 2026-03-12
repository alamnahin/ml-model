import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# Create a single comprehensive diagram with better layout
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 3, 0.5], width_ratios=[1, 1], 
                      hspace=0.15, wspace=0.1)

# Title
title_ax = fig.add_subplot(gs[0, :])
title_ax.text(0.5, 0.5, 'System Architecture: Centralized → Decentralized Credit Scoring\nBlockchain-Based Machine Learning Approach', 
              ha='center', va='center', fontsize=22, fontweight='bold', transform=title_ax.transAxes)
title_ax.axis('off')

# Left subplot - Centralized
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_title('TRADITIONAL CENTRALIZED APPROACH', fontsize=16, fontweight='bold', pad=20, color='#E74C3C')
ax1.axis('off')

# Right subplot - Decentralized  
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_title('PROPOSED DECENTRALIZED APPROACH', fontsize=16, fontweight='bold', pad=20, color='#2ECC71')
ax2.axis('off')

# Enhanced color scheme
colors = {
    'centralized_bad': '#E74C3C',
    'decentralized_good': '#2ECC71',
    'blockchain': '#3498DB',
    'ml_best': '#27AE60',
    'ml_good': '#3498DB',
    'data': '#95A5A6',
    'warning': '#E67E22',
    'user': '#9B59B6'
}

# ============================================================================
# LEFT SIDE: CENTRALIZED SYSTEM (Cleaner Layout)
# ============================================================================

# Title section
ax1.text(5, 9.5, 'CENTRALIZED CREDIT SCORING', ha='center', va='center', 
         fontsize=14, fontweight='bold', color=colors['centralized_bad'])

# Users
user_box1 = FancyBboxPatch((2, 8.5), 6, 0.8, boxstyle="round,pad=0.1", 
                           facecolor=colors['user'], edgecolor='black', linewidth=2)
ax1.add_patch(user_box1)
ax1.text(5, 8.9, 'Credit Applicants', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Data sources
sources = ['Credit Bureaus', 'Bank Records', 'Income Data']
for i, source in enumerate(sources):
    x_pos = 1 + i * 2.7
    box = FancyBboxPatch((x_pos, 7.2), 2.5, 0.6, boxstyle="round,pad=0.05", 
                         facecolor=colors['data'], edgecolor='black', linewidth=1)
    ax1.add_patch(box)
    ax1.text(x_pos + 1.25, 7.5, source, ha='center', va='center', fontsize=10, fontweight='bold')

# Central authority
central = FancyBboxPatch((1.5, 5.5), 7, 1.2, boxstyle="round,pad=0.1", 
                         facecolor=colors['centralized_bad'], edgecolor='black', linewidth=3)
ax1.add_patch(central)
ax1.text(5, 6.1, 'CENTRAL AUTHORITY\n(Banks/Credit Agencies)', ha='center', va='center', 
         fontsize=13, fontweight='bold', color='white')

# Traditional models
trad_model = FancyBboxPatch((2.5, 4.2), 5, 0.8, boxstyle="round,pad=0.05", 
                            facecolor=colors['warning'], edgecolor='black', linewidth=2)
ax1.add_patch(trad_model)
ax1.text(5, 4.6, 'Traditional Credit Score', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='white')

# Database
db = FancyBboxPatch((2, 3), 6, 0.8, boxstyle="round,pad=0.05", 
                    facecolor=colors['data'], edgecolor='black', linewidth=2)
ax1.add_patch(db)
ax1.text(5, 3.4, 'Private Database', ha='center', va='center', 
         fontsize=11, fontweight='bold')

# Decision
decision1 = FancyBboxPatch((2.5, 1.8), 5, 0.8, boxstyle="round,pad=0.05", 
                           facecolor=colors['centralized_bad'], edgecolor='black', linewidth=2)
ax1.add_patch(decision1)
ax1.text(5, 2.2, 'Opaque Decision', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='white')

# Problems list
problems = FancyBboxPatch((0.5, 0.3), 9, 1.2, boxstyle="round,pad=0.1", 
                          facecolor='#FFE5E5', edgecolor='red', linewidth=2)
ax1.add_patch(problems)
ax1.text(5, 0.9, '❌ PROBLEMS\n• Single point of failure  • Opaque process\n• Limited data sources  • Privacy issues', 
         ha='center', va='center', fontsize=10, fontweight='bold', color='red')

# ============================================================================
# RIGHT SIDE: DECENTRALIZED SYSTEM (Our Solution)
# ============================================================================

# Title section
ax2.text(5, 9.5, 'BLOCKCHAIN-BASED CREDIT SCORING', ha='center', va='center', 
         fontsize=14, fontweight='bold', color=colors['decentralized_good'])

# Users
user_box2 = FancyBboxPatch((2, 8.5), 6, 0.8, boxstyle="round,pad=0.1", 
                           facecolor=colors['user'], edgecolor='black', linewidth=2)
ax2.add_patch(user_box2)
ax2.text(5, 8.9, 'Blockchain Users', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Blockchain data sources
bc_sources = ['Ethereum Tx', 'Smart Contracts', 'DeFi Data']
for i, source in enumerate(bc_sources):
    x_pos = 1 + i * 2.7
    box = FancyBboxPatch((x_pos, 7.2), 2.5, 0.6, boxstyle="round,pad=0.05", 
                         facecolor=colors['blockchain'], edgecolor='black', linewidth=1)
    ax2.add_patch(box)
    ax2.text(x_pos + 1.25, 7.5, source, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Blockchain network
network = FancyBboxPatch((0.5, 6.2), 9, 0.8, boxstyle="round,pad=0.1", 
                         facecolor=colors['blockchain'], edgecolor='black', linewidth=3)
ax2.add_patch(network)
ax2.text(5, 6.6, 'DECENTRALIZED BLOCKCHAIN NETWORK', ha='center', va='center', 
         fontsize=13, fontweight='bold', color='white')

# Our ML models section (HIGHLIGHTED)
ml_section = FancyBboxPatch((0.3, 4.5), 9.4, 1.5, boxstyle="round,pad=0.1", 
                            facecolor='#E8F8F5', edgecolor=colors['ml_best'], linewidth=4)
ax2.add_patch(ml_section)

# Individual models
models = [
    {'name': 'RandomForest', 'score': '93.9%', 'label': '🏆', 'color': colors['ml_best']},
    {'name': 'XGBoost', 'score': '93.1%', 'label': '🥈', 'color': colors['ml_good']},
    {'name': 'LightGBM', 'score': '92.7%', 'label': '🥉', 'color': colors['ml_good']},
    {'name': 'AdaBoost', 'score': '91.1%', 'label': '📊', 'color': colors['ml_good']}
]

for i, model in enumerate(models):
    x_pos = 0.7 + i * 2.15
    box = FancyBboxPatch((x_pos, 4.8), 2, 0.9, boxstyle="round,pad=0.05", 
                         facecolor=model['color'], edgecolor='black', linewidth=2)
    ax2.add_patch(box)
    ax2.text(x_pos + 1, 5.25, f"{model['label']} {model['name']}\n{model['score']}", 
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax2.text(5, 4.6, 'OUR ML MODELS (Gas Usage Prediction)', ha='center', va='center', 
         fontsize=12, fontweight='bold', color=colors['ml_best'])

# Data processing
processing = FancyBboxPatch((1, 3.5), 8, 0.6, boxstyle="round,pad=0.05", 
                            facecolor=colors['data'], edgecolor='black', linewidth=2)
ax2.add_patch(processing)
ax2.text(5, 3.8, '1.1M+ Transactions | 5-Fold CV | Data Leakage Prevention', 
         ha='center', va='center', fontsize=10, fontweight='bold')

# Smart contract
smart_contract = FancyBboxPatch((1.5, 2.3), 7, 0.8, boxstyle="round,pad=0.05", 
                                facecolor=colors['decentralized_good'], edgecolor='black', linewidth=3)
ax2.add_patch(smart_contract)
ax2.text(5, 2.7, 'SMART CONTRACT DECISION', ha='center', va='center', 
         fontsize=12, fontweight='bold', color='white')

# Benefits list
benefits = FancyBboxPatch((0.5, 0.3), 9, 1.2, boxstyle="round,pad=0.1", 
                          facecolor='#E8F8F5', edgecolor='green', linewidth=2)
ax2.add_patch(benefits)
ax2.text(5, 0.9, '✅ BENEFITS\n• Decentralized & transparent  • No single failure point\n• Rich blockchain data  • Automated & fair', 
         ha='center', va='center', fontsize=10, fontweight='bold', color='green')

# Add evolution arrow
arrow_ax = fig.add_subplot(gs[2, :])
arrow_ax.annotate('', xy=(0.8, 0.5), xytext=(0.2, 0.5), 
                  arrowprops=dict(arrowstyle='->', lw=8, color='red'),
                  transform=arrow_ax.transAxes)
arrow_ax.text(0.5, 0.7, 'EVOLUTION TO BLOCKCHAIN', ha='center', va='center', 
              fontsize=16, fontweight='bold', color='red', transform=arrow_ax.transAxes)
arrow_ax.text(0.5, 0.3, 'Enhanced Transparency • Better Data • ML Excellence', ha='center', va='center', 
              fontsize=12, fontweight='bold', color='blue', transform=arrow_ax.transAxes)
arrow_ax.axis('off')

# Add flow arrows within each side
# Left side
for y_from, y_to in [(8.4, 7.9), (6.9, 6.8), (5.4, 5.1), (4.1, 3.9), (2.9, 2.7)]:
    ax1.annotate('', xy=(5, y_to), xytext=(5, y_from), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Right side  
for y_from, y_to in [(8.4, 7.9), (6.9, 6.9), (6.1, 6.0), (4.4, 4.2), (3.4, 3.2)]:
    ax2.annotate('', xy=(5, y_to), xytext=(5, y_from), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

plt.savefig('/home/dodopc/Desktop/ml-model/system_architecture_final.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("✅ Final Clean System Architecture Diagram Created!")
print("Generated: system_architecture_final.png") 
print("Key improvements:")
print("• Professional layout with clear sections")
print("• Larger, readable fonts throughout")
print("• Better color contrast and spacing")
print("• Prominent highlighting of our ML research")
print("• Clear evolution arrow and messaging")
print("• Publication-ready quality")

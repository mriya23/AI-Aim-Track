#!/usr/bin/env python3
"""
Aimbot Configuration Optimizer
Helps find optimal settings for different movement scenarios
"""

import json
import itertools
from typing import Dict, List, Tuple
from test_aimbot_performance import AimbotTester, MovementTest

class ConfigurationOptimizer:
    def __init__(self):
        self.tester = AimbotTester()
        self.best_config = None
        self.best_score = 0
        
    def generate_config_variations(self, base_config: Dict) -> List[Dict]:
        """Generate different configuration variations to test"""
        variations = []
        
        # Key parameters to optimize
        prediction_factors = [0.15, 0.20, 0.25, 0.30]
        pid_params = [
            {"kp": 0.8, "ki": 0.06, "kd": 0.15},
            {"kp": 0.95, "ki": 0.08, "kd": 0.20},
            {"kp": 1.1, "ki": 0.10, "kd": 0.25},
            {"kp": 1.2, "ki": 0.12, "kd": 0.30}
        ]
        velocity_smoothing = [0.4, 0.6, 0.8]
        sticky_multipliers = [1.3, 1.5, 1.7, 2.0]
        
        # Generate combinations
        for pred_factor, pid, smooth, sticky in itertools.product(
            prediction_factors, pid_params, velocity_smoothing, sticky_multipliers
        ):
            config = base_config.copy()
            config.update({
                "prediction_max": pred_factor,
                "pid_kp": pid["kp"],
                "pid_ki": pid["ki"],
                "pid_kd": pid["kd"],
                "velocity_smoothing": smooth,
                "lock_sticky_multiplier": sticky
            })
            variations.append(config)
        
        return variations[:20]  # Limit to prevent excessive testing
    
    def evaluate_config(self, config: Dict, test_scenarios: List[MovementTest]) -> float:
        """Evaluate a configuration against test scenarios"""
        total_score = 0
        
        for test in test_scenarios:
            # Run test with current config
            result = self.tester.run_movement_test(test, config)
            
            # Calculate weighted score based on difficulty
            difficulty_multiplier = {
                "Easy": 1.0,
                "Medium": 1.2,
                "Hard": 1.5,
                "Very Hard": 2.0
            }
            
            # Score based on tracking accuracy and distance
            accuracy_score = result["tracking_accuracy"] * difficulty_multiplier.get(test.difficulty, 1.0)
            distance_penalty = max(0, (result["avg_distance"] - 20) * 0.5)  # Penalty for high distance
            
            scenario_score = accuracy_score - distance_penalty
            total_score += scenario_score
        
        return total_score / len(test_scenarios)
    
    def optimize_configuration(self, base_config: Dict, target_scenarios: List[str] = None) -> Dict:
        """Find optimal configuration for specified scenarios"""
        print("Starting configuration optimization...")
        
        # Define test scenarios
        all_scenarios = [
            MovementTest(
                name="Fast Strafe",
                description="High-speed strafe movement",
                movement_func=lambda: self.tester.generate_strafe_movement(speed=500, direction_change_interval=0.3),
                duration=2.0,
                difficulty="Hard"
            ),
            MovementTest(
                name="Sprint Strafe",
                description="Maximum speed strafe movement",
                movement_func=lambda: self.tester.generate_sprint_strafe_movement(speed=600),
                duration=2.0,
                difficulty="Very Hard"
            ),
            MovementTest(
                name="Zigzag Movement",
                description="Diagonal zigzag movement pattern",
                movement_func=lambda: self.tester.generate_zigzag_movement(speed=400, zigzag_frequency=2),
                duration=2.0,
                difficulty="Hard"
            ),
            MovementTest(
                name="Jump Strafe",
                description="Jumping while strafing",
                movement_func=lambda: self.tester.generate_jump_strafe_movement(speed=350),
                duration=2.0,
                difficulty="Hard"
            )
        ]
        
        # Filter scenarios if specified
        if target_scenarios:
            test_scenarios = [s for s in all_scenarios if s.name in target_scenarios]
        else:
            test_scenarios = all_scenarios
        
        print(f"Testing {len(test_scenarios)} scenarios...")
        
        # Generate configuration variations
        config_variations = self.generate_config_variations(base_config)
        print(f"Testing {len(config_variations)} configuration variations...")
        
        best_score = 0
        best_config = None
        
        # Test each configuration
        for i, config in enumerate(config_variations):
            print(f"Testing configuration {i+1}/{len(config_variations)}...")
            
            score = self.evaluate_config(config, test_scenarios)
            print(f"  Score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
                print(f"  New best configuration found! Score: {score:.2f}")
        
        print(f"\nOptimization complete!")
        print(f"Best score: {best_score:.2f}")
        
        return best_config
    
    def generate_optimization_report(self, optimized_config: Dict, base_config: Dict) -> str:
        """Generate optimization report"""
        report = []
        report.append("=" * 60)
        report.append("AIMBOT CONFIGURATION OPTIMIZATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("Key Optimizations:")
        
        # Compare key parameters
        key_params = [
            "prediction_max", "pid_kp", "pid_ki", "pid_kd", 
            "velocity_smoothing", "lock_sticky_multiplier"
        ]
        
        for param in key_params:
            if param in optimized_config and param in base_config:
                old_val = base_config[param]
                new_val = optimized_config[param]
                change = "↑" if new_val > old_val else "↓" if new_val < old_val else "="
                report.append(f"  {param}: {old_val} → {new_val} {change}")
        
        report.append("")
        report.append("Optimized Configuration:")
        report.append(json.dumps(optimized_config, indent=2))
        report.append("")
        
        report.append("Recommendations:")
        report.append("  • Test the optimized configuration in actual gameplay")
        report.append("  • Fine-tune individual parameters based on personal preference")
        report.append("  • Monitor performance and adjust as needed")
        
        return "\n".join(report)

def main():
    """Main optimization function"""
    print("Aimbot Configuration Optimizer")
    print("This will find optimal settings for sticky aim on fast-moving targets")
    print("=" * 60)
    
    # Load current configuration
    try:
        with open("lib/config/config.json", "r") as f:
            base_config = json.load(f)
        print("Loaded current configuration")
    except FileNotFoundError:
        print("Using default base configuration")
        base_config = {
            "prediction_max": 0.25,
            "pid_kp": 0.95,
            "pid_ki": 0.08,
            "pid_kd": 0.20,
            "velocity_smoothing": 0.6,
            "lock_sticky_multiplier": 1.5
        }
    
    # Get optimization targets
    print("\nSelect optimization targets:")
    print("1. Fast Strafe (recommended for competitive play)")
    print("2. Sprint Strafe (maximum speed)")
    print("3. Zigzag Movement")
    print("4. Jump Strafe")
    print("5. All scenarios (comprehensive)")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    target_scenarios = {
        "1": ["Fast Strafe"],
        "2": ["Sprint Strafe"],
        "3": ["Zigzag Movement"],
        "4": ["Jump Strafe"],
        "5": None  # All scenarios
    }.get(choice, None)
    
    # Run optimization
    optimizer = ConfigurationOptimizer()
    optimized_config = optimizer.optimize_configuration(base_config, target_scenarios)
    
    # Generate and display report
    report = optimizer.generate_optimization_report(optimized_config, base_config)
    print(report)
    
    # Save optimized configuration
    save_choice = input("\nSave optimized configuration? (y/n): ").strip().lower()
    if save_choice == 'y':
        # Backup original
        with open("lib/config/config_backup.json", "w") as f:
            json.dump(base_config, f, indent=2)
        
        # Save optimized
        with open("lib/config/config.json", "w") as f:
            json.dump(optimized_config, f, indent=2)
        
        print("Configuration saved!")
        print("Original configuration backed up to config_backup.json")
    
    print("\nOptimization complete!")

if __name__ == "__main__":
    main()
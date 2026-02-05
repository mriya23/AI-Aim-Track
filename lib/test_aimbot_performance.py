#!/usr/bin/env python3
"""
Aimbot Performance Testing Suite
Tests various movement scenarios to validate aimbot improvements
"""

import time
import math
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class MovementTest:
    name: str
    description: str
    movement_func: callable
    duration: float
    difficulty: str

class AimbotTester:
    def __init__(self):
        self.test_results = []
        
    def generate_strafe_movement(self, speed: float = 300, direction_change_interval: float = 0.5) -> List[Tuple[float, float]]:
        """Generate strafe left-right movement pattern"""
        positions = []
        direction = 1
        last_change = 0
        
        for t in range(int(self.duration * 60)):  # 60 FPS
            current_time = t / 60
            if current_time - last_change > direction_change_interval:
                direction *= -1
                last_change = current_time
            
            x = 960 + (direction * speed * (current_time - last_change))
            y = 540 + random.uniform(-10, 10)  # Small vertical jitter
            positions.append((x, y))
            
        return positions
    
    def generate_zigzag_movement(self, speed: float = 400, zigzag_frequency: float = 2) -> List[Tuple[float, float]]:
        """Generate zigzag diagonal movement pattern"""
        positions = []
        
        for t in range(int(self.duration * 60)):
            time_sec = t / 60
            
            # Diagonal movement with periodic direction changes
            x = 960 + (speed * time_sec * 0.3) + (speed * 0.2 * math.sin(zigzag_frequency * time_sec * 2 * math.pi))
            y = 540 + (speed * time_sec * 0.2) + (speed * 0.15 * math.cos(zigzag_frequency * time_sec * 2 * math.pi))
            
            positions.append((x, y))
            
        return positions
    
    def generate_sprint_strafe_movement(self, speed: float = 600) -> List[Tuple[float, float]]:
        """Generate high-speed sprint strafe movement"""
        positions = []
        direction = 1
        last_change = 0
        
        for t in range(int(self.duration * 60)):
            current_time = t / 60
            
            # Rapid direction changes for sprint strafe
            if current_time - last_change > 0.3:
                direction *= -1
                last_change = current_time
            
            x = 960 + (direction * speed * (current_time - last_change))
            y = 540 + random.uniform(-5, 5)
            positions.append((x, y))
            
        return positions
    
    def generate_jump_strafe_movement(self, speed: float = 350) -> List[Tuple[float, float]]:
        """Generate jump + strafe movement pattern"""
        positions = []
        direction = 1
        jump_phase = 0
        
        for t in range(int(self.duration * 60)):
            current_time = t / 60
            
            # Strafe direction changes
            if int(current_time * 2) % 2 == 0:
                direction = 1
            else:
                direction = -1
            
            # Jump pattern (parabolic)
            jump_cycle = (current_time * 1.5) % 1
            if jump_cycle < 0.7:  # Jumping phase
                jump_height = 100 * math.sin(jump_cycle * math.pi)
            else:  # Landing phase
                jump_height = 0
            
            x = 960 + (direction * speed * (current_time % 0.5))
            y = 540 - jump_height + random.uniform(-3, 3)
            positions.append((x, y))
            
        return positions
    
    def generate_random_movement(self, max_speed: float = 500) -> List[Tuple[float, float]]:
        """Generate random movement pattern"""
        positions = []
        current_x, current_y = 960, 540
        
        for t in range(int(self.duration * 60)):
            # Random velocity changes
            vx = random.uniform(-max_speed, max_speed) * 0.1
            vy = random.uniform(-max_speed, max_speed) * 0.05
            
            current_x += vx / 60  # Convert to position change
            current_y += vy / 60
            
            # Keep within reasonable bounds
            current_x = max(800, min(1120, current_x))
            current_y = max(400, min(680, current_y))
            
            positions.append((current_x, current_y))
            
        return positions
    
    def calculate_tracking_metrics(self, target_positions: List[Tuple[float, float]], 
                                 aimbot_responses: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate tracking performance metrics"""
        if len(target_positions) != len(aimbot_responses):
            return {"error": "Mismatched position counts"}
        
        distances = []
        response_times = []
        max_deviations = []
        
        for i, (target, aimbot) in enumerate(zip(target_positions, aimbot_responses)):
            distance = math.dist(target, aimbot)
            distances.append(distance)
            
            # Calculate response time (simplified)
            if i > 0:
                prev_distance = math.dist(target_positions[i-1], aimbot_responses[i-1])
                if distance < prev_distance * 0.8:  # Significant improvement
                    response_times.append(i / 60)  # Response time in seconds
        
        return {
            "avg_distance": sum(distances) / len(distances),
            "max_distance": max(distances),
            "min_distance": min(distances),
            "std_deviation": math.sqrt(sum(d*d for d in distances) / len(distances) - (sum(distances)/len(distances))**2),
            "tracking_accuracy": (1 - (sum(distances) / len(distances)) / 200) * 100,  # Percentage
            "response_time": min(response_times) if response_times else 0,
            "total_frames": len(target_positions)
        }
    
    def simulate_aimbot_response(self, target_positions: List[Tuple[float, float]], 
                               aimbot_config: Dict) -> List[Tuple[float, float]]:
        """Simulate aimbot response to target movement"""
        responses = []
        current_aim_x, current_aim_y = 960, 540
        
        # Simulate PID controller response
        kp = aimbot_config.get("pid_kp", 0.95)
        ki = aimbot_config.get("pid_ki", 0.08)
        kd = aimbot_config.get("pid_kd", 0.20)
        
        integral_x = integral_y = 0
        prev_error_x = prev_error_y = 0
        
        for target_x, target_y in target_positions:
            # Calculate error
            error_x = target_x - current_aim_x
            error_y = target_y - current_aim_y
            
            # PID calculation
            integral_x += error_x * (1/60)
            integral_y += error_y * (1/60)
            
            derivative_x = (error_x - prev_error_x) * 60
            derivative_y = (error_y - prev_error_y) * 60
            
            output_x = kp * error_x + ki * integral_x + kd * derivative_x
            output_y = kp * error_y + ki * integral_y + kd * derivative_y
            
            # Apply movement limits and scaling
            scale = aimbot_config.get("scale", 31)
            sensitivity = aimbot_config.get("smoothing", 0.11)
            
            move_x = output_x * scale / 100 * sensitivity
            move_y = output_y * scale / 100 * sensitivity
            
            # Cap movement
            max_move = 60
            move_x = max(-max_move, min(max_move, move_x))
            move_y = max(-max_move, min(max_move, move_y))
            
            current_aim_x += move_x
            current_aim_y += move_y
            
            responses.append((current_aim_x, current_aim_y))
            
            prev_error_x = error_x
            prev_error_y = error_y
        
        return responses
    
    def run_movement_test(self, test: MovementTest, aimbot_config: Dict) -> Dict:
        """Run a single movement test"""
        print(f"Running test: {test.name} ({test.difficulty})")
        print(f"Description: {test.description}")
        
        # Generate target movement
        target_positions = test.movement_func()
        
        # Simulate aimbot response
        aimbot_responses = self.simulate_aimbot_response(target_positions, aimbot_config)
        
        # Calculate metrics
        metrics = self.calculate_tracking_metrics(target_positions, aimbot_responses)
        metrics["test_name"] = test.name
        metrics["difficulty"] = test.difficulty
        
        print(f"Average tracking distance: {metrics['avg_distance']:.2f} pixels")
        print(f"Tracking accuracy: {metrics['tracking_accuracy']:.1f}%")
        print(f"Max deviation: {metrics['max_distance']:.2f} pixels")
        print("-" * 50)
        
        return metrics
    
    def run_all_tests(self, aimbot_config: Dict) -> List[Dict]:
        """Run all movement tests"""
        tests = [
            MovementTest(
                name="Basic Strafe",
                description="Left-right strafe movement at moderate speed",
                movement_func=lambda: self.generate_strafe_movement(speed=300),
                duration=3.0,
                difficulty="Medium"
            ),
            MovementTest(
                name="Fast Strafe",
                description="High-speed strafe movement with rapid direction changes",
                movement_func=lambda: self.generate_strafe_movement(speed=500, direction_change_interval=0.3),
                duration=3.0,
                difficulty="Hard"
            ),
            MovementTest(
                name="Sprint Strafe",
                description="Maximum speed strafe movement (sprint + strafe)",
                movement_func=lambda: self.generate_sprint_strafe_movement(speed=600),
                duration=2.0,
                difficulty="Very Hard"
            ),
            MovementTest(
                name="Zigzag Movement",
                description="Diagonal zigzag movement pattern",
                movement_func=lambda: self.generate_zigzag_movement(speed=400, zigzag_frequency=2),
                duration=3.0,
                difficulty="Hard"
            ),
            MovementTest(
                name="Jump Strafe",
                description="Jumping while strafing (common in competitive play)",
                movement_func=lambda: self.generate_jump_strafe_movement(speed=350),
                duration=3.0,
                difficulty="Hard"
            ),
            MovementTest(
                name="Random Movement",
                description="Unpredictable random movement",
                movement_func=lambda: self.generate_random_movement(max_speed=500),
                duration=3.0,
                difficulty="Very Hard"
            )
        ]
        
        results = []
        for test in tests:
            result = self.run_movement_test(test, aimbot_config)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict]) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 60)
        report.append("AIMBOT PERFORMANCE TEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        avg_accuracy = sum(r['tracking_accuracy'] for r in results) / len(results)
        avg_distance = sum(r['avg_distance'] for r in results) / len(results)
        
        report.append(f"Overall Performance:")
        report.append(f"  Average Tracking Accuracy: {avg_accuracy:.1f}%")
        report.append(f"  Average Tracking Distance: {avg_distance:.2f} pixels")
        report.append("")
        
        # Detailed results by difficulty
        difficulties = {}
        for result in results:
            diff = result['difficulty']
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(result)
        
        for difficulty, diff_results in difficulties.items():
            report.append(f"{difficulty} Difficulty Tests:")
            for result in diff_results:
                report.append(f"  {result['test_name']}: {result['tracking_accuracy']:.1f}% accuracy, "
                            f"{result['avg_distance']:.1f}px avg distance")
            report.append("")
        
        # Performance analysis
        report.append("Performance Analysis:")
        if avg_accuracy > 85:
            report.append("  ✓ EXCELLENT: Aimbot maintains strong lock on fast-moving targets")
        elif avg_accuracy > 70:
            report.append("  ✓ GOOD: Aimbot performs well with room for minor improvements")
        elif avg_accuracy > 55:
            report.append("  ⚠ FAIR: Aimbot struggles with some fast movement patterns")
        else:
            report.append("  ⚠ NEEDS IMPROVEMENT: Aimbot frequently loses fast-moving targets")
        
        report.append("")
        report.append("Recommendations:")
        if avg_distance > 30:
            report.append("  • Consider increasing prediction factors for faster targets")
            report.append("  • Adjust PID parameters for more aggressive response")
        if any(r['max_distance'] > 100 for r in results):
            report.append("  • Implement better interpolation for temporarily lost targets")
            report.append("  • Increase lock grace time for better target retention")
        
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """Main test function"""
    print("Starting Aimbot Performance Testing Suite")
    print("This will test the aimbot against various challenging movement patterns")
    print("=" * 60)
    
    # Load current aimbot configuration
    import json
    try:
        with open("lib/config/config.json", "r") as f:
            aimbot_config = json.load(f)
        print("Loaded current aimbot configuration")
    except FileNotFoundError:
        print("Using default configuration")
        aimbot_config = {
            "pid_kp": 0.95,
            "pid_ki": 0.08,
            "pid_kd": 0.20,
            "scale": 31,
            "smoothing": 0.11,
            "prediction_max": 0.25,
            "prediction_min": 0.05
        }
    
    # Run tests
    tester = AimbotTester()
    results = tester.run_all_tests(aimbot_config)
    
    # Generate and display report
    report = tester.generate_report(results)
    print(report)
    
    # Save results
    with open("aimbot_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to aimbot_test_results.json")
    print("Testing complete!")

if __name__ == "__main__":
    main()
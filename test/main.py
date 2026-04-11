import robot_detection
import robot_fingerprinting

artifacts = robot_detection.execute(previous_artifacts={})
artifacts = robot_fingerprinting.execute(artifacts)

fingerprints = artifacts["fingerprints"]

for k, v in fingerprints.items():
    print(f"\nTeam {k}: {v}\n")
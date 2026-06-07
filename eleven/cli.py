"""
Command-line interface for Eleven.

Usage:
    python -m eleven.cli test          # Test with simulated data
    python -m eleven.cli connect       # Test real Muse connection
    python -m eleven.cli server        # Run the WebSocket server
    python -m eleven.cli calibrate     # Run calibration session
"""

import argparse
import logging
import time
import sys

import numpy as np


def cmd_test(args):
    """Test the full pipeline with simulated data."""
    from eleven.acquisition import SimulatedMuse
    from eleven.preprocessing import EEGPreprocessor
    from eleven.artifacts import ArtifactDetector, ControlSignal
    from eleven.features import AttentionEstimator

    print("=" * 60)
    print("Eleven Pipeline Test (Simulated Data)")
    print("=" * 60)
    print()

    muse = SimulatedMuse()
    preprocessor = EEGPreprocessor()
    detector = ArtifactDetector()
    attention = AttentionEstimator()

    muse.connect()
    muse.start_stream()

    print("Running for 10 seconds...")
    print("Simulated blinks will be injected randomly.")
    print()

    signal_counts = {s.value: 0 for s in ControlSignal}
    attention_samples = []

    start_time = time.time()
    while time.time() - start_time < 10:
        data = muse.get_data(64)
        if data is not None:
            windows = preprocessor.process(data)

            for window in windows:
                timestamp = time.time()

                # Artifact detection
                signal = detector.process(window, timestamp)
                if signal is not None:
                    signal_counts[signal.value] += 1
                    if signal != ControlSignal.NONE:
                        print(f"  >> DETECTED: {signal.value}")

                # Attention estimation
                metrics = attention.estimate(window)
                attention_samples.append(metrics)

        time.sleep(0.1)

    muse.disconnect()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    print("Signals detected:")
    for signal, count in signal_counts.items():
        if count > 0:
            print(f"  {signal}: {count}")

    if attention_samples:
        avg_focus = np.mean([s["focus"] for s in attention_samples])
        avg_relax = np.mean([s["relaxation"] for s in attention_samples])
        print()
        print(f"Average focus: {avg_focus:.2f}")
        print(f"Average relaxation: {avg_relax:.2f}")

    print()
    print("Test complete!")


def cmd_connect(args):
    """Test real Muse connection."""
    from eleven.acquisition import MuseAcquisition

    print("=" * 60)
    print("Muse S Connection Test")
    print("=" * 60)
    print()
    print("Ensure your Muse S is:")
    print("  1. Turned on")
    print("  2. Not connected to another device")
    print("  3. Within Bluetooth range")
    print()

    muse = MuseAcquisition()

    print("Connecting...")
    if not muse.connect():
        print("Failed to connect. Check that Muse is on and nearby.")
        return 1

    print("Connected! Starting stream...")
    muse.start_stream()

    print("Streaming for 5 seconds...")
    for i in range(5):
        time.sleep(1)
        data = muse.get_data()
        if data is not None:
            eeg = data[:4, :]
            print(f"  Second {i+1}: {eeg.shape[1]} samples, "
                  f"range: [{eeg.min():.1f}, {eeg.max():.1f}] μV")

    muse.disconnect()
    print()
    print("Connection test successful!")
    return 0


def cmd_server(args):
    """Run the WebSocket server."""
    from eleven.server import main
    main()


def cmd_calibrate(args):
    """Run interactive calibration session."""
    from eleven.acquisition import create_acquisition
    from eleven.preprocessing import EEGPreprocessor
    from eleven.vq_encoder import EEGTokenizer

    print("=" * 60)
    print("Eleven Calibration Session")
    print("=" * 60)
    print()

    use_sim = args.simulate
    muse = create_acquisition(use_simulation=use_sim)
    preprocessor = EEGPreprocessor()
    tokenizer = EEGTokenizer()

    if not muse.connect():
        print("Failed to connect to Muse")
        return 1

    muse.start_stream()
    training_data = {}

    intents = ["rest", "blink", "focus", "relax"]

    for intent in intents:
        print()
        print(f"Step: {intent.upper()}")
        if intent == "rest":
            print("Sit still and relax. Look at the screen neutrally.")
        elif intent == "blink":
            print("Blink deliberately (strong blinks) every 2 seconds.")
        elif intent == "focus":
            print("Concentrate intensely. Do mental math or focus on a point.")
        elif intent == "relax":
            print("Close your eyes and relax. Think of something calming.")

        input("Press Enter when ready...")
        print("Recording for 10 seconds...")

        windows = []
        start = time.time()
        while time.time() - start < 10:
            data = muse.get_data(64)
            if data is not None:
                processed = preprocessor.process(data)
                windows.extend(processed)
            time.sleep(0.1)

        training_data[intent] = windows
        print(f"  Collected {len(windows)} windows for '{intent}'")

    print()
    print("Calibrating tokenizer...")
    tokenizer.calibrate(training_data)

    # Save calibration
    from pathlib import Path
    save_dir = Path("calibration_data")
    save_dir.mkdir(exist_ok=True)
    tokenizer.save(save_dir)
    print(f"Saved calibration to {save_dir}/")

    muse.disconnect()
    print()
    print("Calibration complete!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Eleven: EEG-based communication platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # test command
    test_parser = subparsers.add_parser("test", help="Test with simulated data")

    # connect command
    connect_parser = subparsers.add_parser("connect", help="Test real Muse connection")

    # server command
    server_parser = subparsers.add_parser("server", help="Run WebSocket server")

    # calibrate command
    calibrate_parser = subparsers.add_parser("calibrate", help="Run calibration session")
    calibrate_parser.add_argument(
        "--simulate", "-s", action="store_true",
        help="Use simulated data instead of real Muse"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.command == "test":
        return cmd_test(args)
    elif args.command == "connect":
        return cmd_connect(args)
    elif args.command == "server":
        return cmd_server(args)
    elif args.command == "calibrate":
        return cmd_calibrate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

import { useState, useCallback, useRef } from "react";
import { useMuseStream } from "./hooks/useMuseStream";
import { EEGDashboard } from "./components/EEGDashboard";
import { Calibration } from "./components/Calibration";
import "./index.css";

export default function App() {
  const [mode, setMode] = useState<"calibration" | "dashboard">("calibration");
  const { status, frame, rlState, brainState, lastAction, learning } =
    useMuseStream();
  const wsRef = useRef<WebSocket | null>(null);

  const sendCommand = useCallback((cmd: object) => {
    // TODO: send calibration commands to server via WS
    console.log("calibration command:", cmd);
  }, []);

  if (mode === "calibration") {
    return (
      <Calibration
        brainState={brainState}
        onComplete={(data) => {
          console.log("Calibration complete:", data);
          setMode("dashboard");
        }}
        sendCommand={sendCommand}
      />
    );
  }

  return <EEGDashboard />;
}

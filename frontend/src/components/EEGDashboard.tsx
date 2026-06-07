import { useMuseStream } from "../hooks/useMuseStream";
import { EEGCanvas } from "./EEGCanvas";
import { BandBars } from "./BandBars";
import { StatePanel } from "./StatePanel";
import { StatusDot } from "./StatusDot";
import { BrainStatePanel } from "./BrainStatePanel";
import { ActionFeed } from "./ActionFeed";

export function EEGDashboard() {
  const { status, frame, rlState, brainState, lastAction, learning } =
    useMuseStream();

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        padding: 24,
        gap: 16,
        background: "#0a0a14",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h1
          style={{
            fontSize: 22,
            fontWeight: 700,
            letterSpacing: 3,
            color: "#00e68a",
            margin: 0,
          }}
        >
          AXIOM · NEURAL INTERFACE
        </h1>
        <StatusDot status={status} sampleCount={frame?.totalSamples ?? 0} />
      </div>

      {/* Main content */}
      <div style={{ display: "flex", flex: 1, gap: 16, minHeight: 0 }}>
        {/* Left column: EEG + Agent */}
        <div
          style={{
            flex: 5,
            display: "flex",
            flexDirection: "column",
            gap: 16,
            minHeight: 0,
          }}
        >
          {/* EEG Waveforms */}
          <div
            style={{
              flex: 3,
              background: "#0d0d1a",
              borderRadius: 12,
              border: "1px solid #1e1e3a",
              overflow: "hidden",
            }}
          >
            <EEGCanvas frame={frame} />
          </div>

          {/* Agent Action Feed */}
          <div
            style={{
              flex: 2,
              background: "#12122a",
              borderRadius: 12,
              border: "1px solid #1e1e3a",
              padding: 16,
              overflow: "hidden",
            }}
          >
            <ActionFeed lastAction={lastAction} learning={learning} />
          </div>
        </div>

        {/* Right sidebar */}
        <div
          style={{
            flex: 2,
            display: "flex",
            flexDirection: "column",
            gap: 16,
            minHeight: 0,
          }}
        >
          {/* Brain State */}
          <div
            style={{
              flex: 2,
              background: "#12122a",
              borderRadius: 12,
              border: "1px solid #1e1e3a",
              padding: 16,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <BrainStatePanel state={brainState} />
          </div>

          {/* Band power */}
          <div
            style={{
              flex: 1,
              background: "#12122a",
              borderRadius: 12,
              border: "1px solid #1e1e3a",
              padding: 16,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <span
              style={{
                fontSize: 11,
                color: "#4a4a6a",
                letterSpacing: 2,
                marginBottom: 12,
              }}
            >
              BAND POWER
            </span>
            <BandBars bands={frame?.bands ?? null} />
          </div>

          {/* RL State */}
          <div
            style={{
              flex: 1,
              background: "#12122a",
              borderRadius: 12,
              border: "1px solid #1e1e3a",
              padding: 16,
              display: "flex",
              flexDirection: "column",
              overflow: "auto",
            }}
          >
            <StatePanel state={rlState} />
          </div>
        </div>
      </div>
    </div>
  );
}

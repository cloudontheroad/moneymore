import { createConnection } from "node:net";

// Vinext builds replace dist in place; a running server keeps the old manifest.
const socket = createConnection({ host: "127.0.0.1", port: 3000 });
socket.once("connect", () => {
  console.error("Dashboard port 3000 is active. Build in a separate checkout or stop the frontend before rebuilding dist.");
  process.exitCode = 1;
  socket.destroy();
});
socket.once("error", (error) => {
  if (error.code !== "ECONNREFUSED") {
    console.error("Unable to verify that dashboard port 3000 is idle:", error.code);
    process.exitCode = 1;
  }
});

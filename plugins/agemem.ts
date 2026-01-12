import { spawn } from "node:child_process";

/**
 * Agemem Plugin: Unified Agentic Memory Management
 * Automatically learns and retrieves facts across sessions.
 */
export const AgememPlugin = async (ctx) => {
  // Path to the agemem python project
  // This value is automatically updated by setup.sh during installation
  const agememPath = process.env.AGEMEM_PATH || "__AGEMEM_PATH__";
  
  const runAgemem = (args) => {
    return new Promise((resolve, reject) => {
      const child = spawn("python3", ["-m", "src.bridge_cli", ...args], {
        cwd: agememPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONPATH: agememPath }
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => (stdout += data));
      child.stderr.on("data", (data) => (stderr += data));

      child.on("close", (code) => {
        if (code === 0) resolve(stdout.trim());
        else reject(new Error(`Agemem failed with code ${code}: ${stderr}`));
      });
    });
  };

  return {
    /**
     * Retrieval: Automatically pull relevant facts into the thinking process.
     */
    "agent.think.before": async (input, output) => {
      const lastMessage = input.messages[input.messages.length - 1]?.content;
      if (!lastMessage) return;
      
      const messageCount = input.messages.length;
      
      try {
        // 1. Semantic search for the current message
        let queries = [runAgemem(["retrieve", lastMessage])];

        // 2. If session is new, also pull "General Profile/Facts"
        if (messageCount <= 2) {
          queries.push(runAgemem(["retrieve", "What are the core facts and identity details about the user?"]));
        }

        const results = await Promise.all(queries);
        const combinedMemories = results
          .filter(r => r && r.trim().length > 0)
          .join("\n---\n");

        if (combinedMemories) {
          output.instructions = (output.instructions || "") + 
            "\n--- AGEMEM LONG-TERM CONTEXT ---\n" +
            "The following facts have been retrieved from your long-term memory. " +
            "Prioritize these facts for personalization and continuity.\n\n" +
            combinedMemories + "\n" +
            "--- END AGEMEM CONTEXT ---\n";
        }
      } catch (err) { console.error("[Agemem] Retrieval Error:", err); }
    },

    /**
     * Learning: Proactively save the interaction to build cross-session memory.
     */
    "agent.think.after": async (input, output) => {
      const lastUserMsg = input.messages[input.messages.length - 1];
      const agentResponse = output.content;
      if (!agentResponse) return;

      // Create a factual record of the turn
      const timestamp = new Date().toISOString();
      const memoryFragment = `[Session Turn ${timestamp}]\nUSER: ${lastUserMsg?.content}\nAGENT: ${agentResponse}`;
      
      try { 
        await runAgemem(["memorize", memoryFragment]); 
      } catch (err) { console.error("[Agemem] Save Error:", err); }
    },

    /**
     * Safe Archiving: Compaction Hook
     * Only save if the conversation is long enough to avoid spamming the database.
     */
    "agent.compaction.before": async (input) => {
      const { messages } = input;
      if (!messages || messages.length < 5) return;
      const content = messages.map((m) => `[${m.role.toUpperCase()}]: ${m.content}`).join("\n---\n");
      try { 
        await runAgemem(["memorize", `ARCHIVED CHUNK:\n${content}`]); 
      } catch (err) {}
    },

    /**
     * Search Tool for the agent
     */
    "tool.execute.before": async (input, output) => {
      if (input.tool === "agemem_search") {
        try {
          output.result = await runAgemem(["retrieve", input.args.query]) || "No relevant memories found.";
          return false;
        } catch (err) { output.result = `Error: ${err.message}`; return false; }
      }
    }
  };
};

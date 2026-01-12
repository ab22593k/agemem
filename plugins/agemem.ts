/**
 * Agemem Plugin: Unified Agentic Memory Management
 * Automatically learns and retrieves facts across sessions.
 */
export const AgememPlugin = async (ctx: any) => {
  // Path to the agemem python project
  // This value is automatically updated by setup.sh during installation
  const agememPath = process.env.AGEMEM_PATH || "__AGEMEM_PATH__";

  const runAgemem = async (args: any) => {
    const proc = Bun.spawn(["python3", "-m", "src.bridge_cli", ...args], {
      cwd: agememPath,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: agememPath,
      },
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    if (exitCode === 0) {
      return stdout.trim();
    } else {
      throw new Error(`Agemem failed with code ${exitCode}: ${stderr}`);
    }
  };

  return {
    /**
     * Retrieval: Automatically pull relevant facts into the thinking process.
     */
    "agent.think.before": async (input: any, output: any) => {
      const lastMessage = input.messages[input.messages.length - 1]?.content;
      if (!lastMessage) return;

      const messageCount = input.messages.length;

      try {
        // Parallel queries to different memory "departments"
        const queries = [
          // 1. Experiential: How have we handled similar intents/tasks before?
          runAgemem(["retrieve", lastMessage, "--experiential", "--links"]),
          // 2. Factual: Who is the user and what are their confirmed traits?
          runAgemem(["retrieve", lastMessage, "--factual", "--links"]),
        ];

        // 3. New Session Bootstrapping: Pull Core Identity explicitly
        if (messageCount <= 2) {
          queries.push(
            runAgemem([
              "retrieve",
              "User core identity, preferences, and active projects",
              "--factual",
              "--links",
            ]),
          );
        }

        const [expResults, factResults, identityResults] =
          await Promise.all(queries);

        let cognitiveContext = "";

        if (factResults || identityResults) {
          cognitiveContext +=
            "\n[FACTUAL ANCHORS - WHO THE USER IS]\n" +
            (factResults || identityResults);
        }

        if (expResults) {
          cognitiveContext +=
            "\n[EXPERIENTIAL TRACES - HOW WE WORK]\n" + expResults;
        }

        if (cognitiveContext.trim()) {
          output.instructions =
            (output.instructions || "") +
            "\n--- COGNITIVE GOVERNANCE (LTM retrieved) ---\n" +
            "Adhere to the following retrieved facts and past experiences to ensure continuity. " +
            "If current instructions contradict these anchors, clarify with the user.\n" +
            cognitiveContext +
            "\n" +
            "--- END COGNITIVE CONTEXT ---\n";
        }
      } catch (err) {
        console.error("[Agemem] Retrieval Error:", err);
      }
    },

    /**
     * Learning: Proactively save the interaction and refine the network.
     * Uses Significance Filtering: Only saves turns that add meaningful value.
     */
    "agent.think.after": async (input: any, output: any) => {
      const lastUserMsg = input.messages[input.messages.length - 1];
      const agentResponse = output.content;
      if (!agentResponse || agentResponse.length < 50) return; // Ignore trivial responses

      // Basic significance check: does it look like a decision, a fact, or a workflow?
      const significanceRegex =
        /decide|confirm|always|never|fact|note|project|update|remember/i;
      const isSignificant =
        significanceRegex.test(agentResponse) ||
        significanceRegex.test(lastUserMsg?.content || "");

      if (!isSignificant && input.messages.length > 4) {
        // Not significant and not a session start - skip noisy memorization
        return;
      }

      const memoryFragment = `[Turn] USER: ${lastUserMsg?.content}\nAGENT: ${agentResponse}`;
      const memoryFunc = isSignificant ? "--factual" : "--experiential";

      try {
        await runAgemem(["memorize", memoryFragment, memoryFunc]);
      } catch (err) {
        console.error("[Agemem] Save Error:", err);
      }
    },

    /**
     * Safe Archiving: Compaction Hook
     * Performs deep maintenance: Pruning and Temporal Decay.
     */
    "agent.compaction.before": async (input: any) => {
      const { messages } = input;
      if (!messages || messages.length < 5) return;

      const content = messages
        .map((m: any) => `[${m.role.toUpperCase()}]: ${m.content}`)
        .join("\n---\n");
      try {
        // 1. Archive the block as EXPERIENTIAL (historical trajectory)
        const archiveResult = await runAgemem([
          "memorize",
          `ARCHIVED SESSION CHUNK:\n${content}`,
          "--experiential",
        ]);
        const survivorId = archiveResult.match(/Success: ([a-f0-9]+)/)?.[1];

        // 2. Perform deep pruning
        await runAgemem(["prune", "ARCHIVED SESSION CHUNK"]);

        // 3. Apply temporal decay
        await runAgemem(["decay", "0.98"]); // Slower decay for more stable associations
      } catch (err) {
        console.error("[Agemem] Compaction Maintenance Error:", err);
      }
    },

    /**
     * Search Tool for the agent
     */
    "tool.execute.before": async (input: any, output: any) => {
      if (input.tool === "agemem_search") {
        try {
          output.result =
            (await runAgemem(["retrieve", input.args.query])) ||
            "No relevant memories found.";
          return false;
        } catch (err: any) {
          output.result = `Error: ${err.message}`;
          return false;
        }
      }
    },
  };
};

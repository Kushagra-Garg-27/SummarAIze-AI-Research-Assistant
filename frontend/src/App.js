import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react"; // icons
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "👋 Hi! I’m your AI research assistant. Ask me anything!",
    },
  ]);
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user message
    setMessages([...messages, { sender: "user", text: input }]);
    const userMessage = input;
    setInput("");

    try {
      // Flask backend call
      const response = await fetch("http://127.0.0.1:8000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: userMessage }),
      });

      const data = await response.json();

      if (data.ok) {
        const { sections } = data;
        let botReply = "";

        if (sections.overview) {
          botReply += `📌 Overview:\n${sections.overview}\n\n`;
        }
        if (sections.features?.length) {
          botReply += `✨ Features:\n${sections.features
            .map((f) => `- ${f}`)
            .join("\n")}\n\n`;
        }
        if (sections.advantages?.length) {
          botReply += `✅ Advantages:\n${sections.advantages
            .map((a) => `- ${a}`)
            .join("\n")}\n\n`;
        }
        if (sections.disadvantages?.length) {
          botReply += `⚠️ Disadvantages:\n${sections.disadvantages
            .map((d) => `- ${d}`)
            .join("\n")}\n\n`;
        }
        if (sections.applications?.length) {
          botReply += `💡 Applications:\n${sections.applications
            .map((app) => `- ${app}`)
            .join("\n")}\n\n`;
        }

        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: botReply.trim() || "No details found." },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: "⚠️ Error: No valid response from backend.",
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "❌ Failed to connect to backend." },
      ]);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -250 }}
            animate={{ x: 0 }}
            exit={{ x: -250 }}
            transition={{ duration: 0.3 }}
            className="w-64 bg-white shadow-md flex flex-col justify-between z-20"
          >
            <div>
              <h2 className="text-xl font-bold p-4 border-b flex justify-between items-center">
                Chats
                <button onClick={() => setSidebarOpen(false)}>
                  <X size={20} className="text-gray-600 hover:text-black" />
                </button>
              </h2>
              <div className="p-4 text-gray-600 space-y-2">
                {["Welcome Chat", "Research Notes"].map((item, idx) => (
                  <motion.p
                    key={idx}
                    whileHover={{ x: 5, scale: 1.02 }}
                    className="cursor-pointer transition-colors duration-200 hover:text-black"
                  >
                    {item}
                  </motion.p>
                ))}
              </div>
            </div>
            <div className="p-4 border-t text-gray-500">Username</div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Section */}
      <div className="flex-1 flex flex-col">
        {/* Top bar with hamburger */}
        <div className="p-4 border-b bg-white shadow-sm flex items-center">
          {!sidebarOpen && (
            <button onClick={() => setSidebarOpen(true)} className="mr-3">
              <Menu size={24} className="text-gray-700 hover:text-black" />
            </button>
          )}
          <h1 className="text-lg font-semibold">AI Assistant</h1>
        </div>

        {/* Messages */}
        <div className="flex-1 p-6 overflow-y-auto space-y-4">
          <AnimatePresence>
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className={`flex ${
                  msg.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`p-3 rounded-2xl max-w-md shadow-md ${
                    msg.sender === "user"
                      ? "bg-blue-500 text-white"
                      : "bg-gray-200 text-gray-800"
                  }`}
                >
                  {msg.text}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {/* Input Box */}
        <div className="p-4 border-t flex bg-white">
          <motion.input
            whileFocus={{ scale: 1.02, borderColor: "#3b82f6" }}
            type="text"
            className="flex-1 border rounded-full px-4 py-2 focus:outline-none transition-all"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <motion.button
            whileTap={{ scale: 0.9 }}
            className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 shadow-md"
            onClick={handleSend}
          >
            ➤
          </motion.button>
        </div>
      </div>
    </div>
  );
}

export default App;

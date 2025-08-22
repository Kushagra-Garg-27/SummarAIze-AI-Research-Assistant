import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "./App.css"; // we'll style it separately

function App() {
  const [topic, setTopic] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!topic.trim()) return;

    setChat([...chat, { sender: "user", text: topic }]);
    setLoading(true);

    try {
      const resp = await axios.post("http://127.0.0.1:8000/summarize", {
        topic: topic,
      });

      const data = resp.data;
      setChat((prev) => [
        ...prev,
        { sender: "bot", text: formatResponse(data) },
      ]);
    } catch (err) {
      setChat((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Error connecting to backend." },
      ]);
    }

    setTopic("");
    setLoading(false);
  };

  const formatResponse = (data) => {
    if (!data.ok) return "❌ No data found.";
    const sections = data.sections;

    let formatted = `📌 **Overview**\n${sections.overview}\n\n`;

    if (sections.features?.length) {
      formatted += "✨ Features:\n";
      sections.features.forEach((f) => (formatted += `- ${f}\n`));
      formatted += "\n";
    }
    if (sections.applications?.length) {
      formatted += "⚙️ Applications:\n";
      sections.applications.forEach((a) => (formatted += `- ${a}\n`));
      formatted += "\n";
    }
    if (sections.advantages?.length) {
      formatted += "✅ Advantages:\n";
      sections.advantages.forEach((adv) => (formatted += `- ${adv}\n`));
      formatted += "\n";
    }
    if (sections.disadvantages?.length) {
      formatted += "⚠️ Disadvantages:\n";
      sections.disadvantages.forEach((d) => (formatted += `- ${d}\n`));
      formatted += "\n";
    }

    if (data.sources?.length) {
      formatted += "🔗 Sources:\n";
      data.sources.forEach((s) => (formatted += `- ${s.name}: ${s.url}\n`));
    }

    return formatted;
  };

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        🤖 SummarAIze Assistant
        <p className="tagline">Your research buddy powered by AI</p>
      </div>

      {/* Chat Area */}
      <div className="chat-box">
        {chat.map((msg, idx) => (
          <div
            key={idx}
            className={`chat-bubble ${msg.sender === "user" ? "user" : "bot"}`}
          >
            <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        ))}

        {loading && <div className="chat-bubble bot">⏳ Thinking...</div>}
      </div>

      {/* Input Area */}
      <div className="chat-input">
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Ask me about a topic..."
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button onClick={handleSend}>🚀</button>
      </div>
    </div>
  );
}

export default App;

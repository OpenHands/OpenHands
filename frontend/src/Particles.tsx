import React, { useEffect } from "react";
import "./theme.css";

export default function Particles() {
  useEffect(() => {
    const container = document.createElement("div");
    container.className = "particles";
    document.body.appendChild(container);

    let running = true;
    const createParticle = () => {
      if (!running) return;
      const p = document.createElement("div");
      p.className = "particle";
      p.style.left = Math.random() * 100 + "vw";
      p.style.top = (-10 + Math.random() * 10) + "vh";
      p.style.animationDuration = 6 + Math.random() * 8 + "s";
      container.appendChild(p);
      setTimeout(() => { p.remove(); }, 14000);
    };

    const interval = setInterval(createParticle, 700);
    for (let i = 0; i < 8; i++) createParticle();

    return () => {
      running = false;
      clearInterval(interval);
      container.remove();
    };
  }, []);

  return null;
}

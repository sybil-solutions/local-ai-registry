"use client";
import { useEffect } from "react";

// Marks each dot picture `.seen` the first time it scrolls into view, which starts its sweep (Dots.module.css).
export default function Seen() {
  useEffect(() => {
    const all = document.querySelectorAll("[data-dots]");
    if (!("IntersectionObserver" in window)) { all.forEach((e) => e.classList.add("seen")); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("seen"); io.unobserve(e.target); } }), { threshold: 0.25 });
    all.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
  return null;
}

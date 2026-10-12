import s from "./Dots.module.css";

// A picture drawn in the dots of the Omarchy hands (art by @samdape): one character per cell, read like ASCII art.
//   #  a full dot      +  a small shading dot      x  a crimson dot, the thing that is happening      space  nothing
// Cells are as wide as a monospace character and about twice as tall, like the hands.
// The dots sweep in from the left once the picture scrolls into view (see Seen.tsx); crimson dots then blink.

const PITCH = 10;

export default function Dots({ art, label }: { art: string[]; label?: string }) {
  const w = Math.max(...art.map((r) => r.length));
  const dy = PITCH * 1.86;
  const dots: React.ReactNode[] = [];
  art.forEach((row, y) => {
    [...row].forEach((ch, x) => {
      if (ch === " ") return;
      const big = ch !== "+";
      const cls = ch === "x" ? s.hot : big ? s.on : s.shade;
      const style = { ["--x" as string]: x, ["--d" as string]: `${((x * 7 + y * 13) % 17) * 0.23}s` } as React.CSSProperties;
      dots.push(<circle key={`${x}.${y}`} cx={x * PITCH + PITCH / 2} cy={y * dy + dy / 2} r={big ? PITCH * 0.36 : PITCH * 0.17} className={cls} style={style} />);
    });
  });
  return (
    <svg viewBox={`0 0 ${w * PITCH} ${art.length * dy}`} className={s.dots} data-dots="" role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      {dots}
    </svg>
  );
}

import Dots from "./Dots";

// The top of a registry page: a title, one line, and a dot picture in the style of the hands.
export default function PageHead({ title, sub, art, children }: { title: string; sub?: string; art?: string[]; children?: React.ReactNode }) {
  return (
    <div className="phead">
      <div className="phead-text">
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
        {children}
      </div>
      {art && <div className="phead-art"><Dots art={art} /></div>}
    </div>
  );
}

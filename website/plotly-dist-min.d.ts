// Minimal stub: the dist-min build at runtime; we rely on @types/plotly.js for the type surface.
declare module 'plotly.js-dist-min' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const Plotly: any;
  export default Plotly;
}

export {};

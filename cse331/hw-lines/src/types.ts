//Used for storing line data and making it portable
export interface ColoredEdge {
    x1: number; // x coordinate of start point
    y1: number; // y coordinate of start point
    x2: number; // x coordinate of end point
    y2: number; // y coordinate of end point
    color: string; // color of line
    key: string // a unique key for the line
}
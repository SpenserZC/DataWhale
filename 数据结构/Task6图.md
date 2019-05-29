# Datawhale 系列数据结构
## Task6  图

图，基本概念：
	![1120165-20180209213532810-1976605857](D:\workspace_kattle\pictures-md\1120165-20180209213532810-1976605857.png)

	1.邻接：如果两个定点同一条边连接，就成这两个定点是邻接的。
	2.路径：路径是边的序列，比如从顶点B到定点J的路径为BAEJ，当然还有别的路径BCDJ。
	3.连通图和非连通图：如果至少一条路径可以连接所有的定点，这个图称为联通图。如果存在从某个定点不能到达另外一个定点，则称为非联通的。
	4.有向图和无向图：如果图中的边没有方向，可以从任意一边到达另一边，则称为无向图。例如从A城市可以到B城市，也可以从B城市到A城市，这叫做无向图。但是如果只能从A城市驶向B城市的图，称为有向图。
	5.有权图和无权图：图中的边呗赋予一个权值，全职是一个数字，代表两个定点间的物理距离，或者从一个顶点到另一个顶点时间，这种图被称作有权图。反之边没有赋权的图被称为无权图

#### 6.1实现有向图、无向图、有权图、无权图的邻接矩阵和邻接表表示方法
```java
//图所需要的，节点，队列，栈

//节点
public class Vertex {
	public char label;
	public boolean wasVisited;
	
	public Vertex(char label){
		this.label = label;
		wasVisited = false;
	}
}	
//队列
public class QueueX {
	private final int SIZE = 20;
    private int[] queArray;
    private int front;
    private int rear;
     
    public QueueX(){
        queArray = new int[SIZE];
        front = 0;
        rear = -1;
    }
     
    public void insert(int j) {
        if(rear == SIZE-1) {
            rear = -1;
        }
        queArray[++rear] = j;
    }
     
    public int remove() {
        int temp = queArray[front++];
        if(front == SIZE) {
            front = 0;
        }
        return temp;
    }
     
    public boolean isEmpty() {
        return (rear+1 == front || front+SIZE-1 == rear);
    }
}

//栈
public class StackX {
	private final int SIZE = 20;
	private int [] st;
	private int top;
	
	public StackX(){
		st = new int[SIZE];
		top=-1;
	}
	public void push(int j){
		st[++top] = j;
	}
	public int pop(){
		return st[top--];
	}
	public int peek(){
		return st[top];
	}
	public boolean isEmpty(){
		return (top == -1);
	}
}
```
```java
//无权无向图，用邻接表表示，
public class Graph {
	private final int MAX_VERTS = 20;//表示定点的个数
	private Vertex vertexList[];//用来存储定点的数组
	private int adjMat[][];//用邻接矩阵来存储边，数组元素0表示没有边界，1表示有边界
	private int nVerts;
	private StackX theStack;//用栈实现深度优先搜多
	private QueueX queue;
	
	public Graph(){
        vertexList = new Vertex[MAX_VERTS];
        adjMat = new int[MAX_VERTS][MAX_VERTS];
        nVerts = 0;//初始化顶点个数为0
        //初始化邻接矩阵所有元素都为0，即所有顶点都没有边
        for(int i = 0; i < MAX_VERTS; i++) {
            for(int j = 0; j < MAX_VERTS; j++) {
                adjMat[i][j] = 0;
            }
        }
        theStack = new StackX();
        queue = new QueueX();
    }
	
	//将顶点添加到数组中，是否访问标志置为wasVisited=false（未访问）
	public void addVertex(char lab){
		vertexList[nVerts++]=new Vertex(lab);
	}
	
	//注意用临界矩阵表示边，是对称的，两部分都要赋值
	public void addEdge(int start,int end){
		adjMat[start][end]=1;
		adjMat[end][start]=1;
	}
	
	//打印某个顶点表示的值
	public void displayVertex(int v){
		System.out.println(vertexList[v].label);
	}
	
	/**深度优先搜索算法
	 * 1.用peek()方法检查栈顶的顶点
	 * 2.用getAdjUnvisitedVertex()方法找到当前栈顶邻接且未被访问的顶点
	 * 3.第二步方法值不等-1则找到下一个未访问的邻接顶点，访问这个顶点，并入栈
	 * 	如果第二步方法返回值等于-1,则没有找到，出栈
	 */
	public void depthFirstSearch(){
		vertexList[0].wasVisited = true;//访问之后标记未true
		displayVertex(0);
		theStack.push(0);
		
		while(!theStack.isEmpty()){
			int v=getAdjUnvisitedVertex(theStack.peek());
			if(v==-1){
				theStack.pop();
			}else{
				vertexList[v].wasVisited = true;
				displayVertex(v);
				theStack.push(v);
			}
		}
		
		for(int i=0;i<nVerts;i++){

			vertexList[i].wasVisited=false;
		}
	}
	
	//找到与某一点邻接且未被访问的顶点
	public int getAdjUnvisitedVertex(int v){
		for(int i=0;i<nVerts;i++){
			if(adjMat[v][i]==1 && vertexList[i].wasVisited == false){
				return i;
			}
		}
		return -1;
	}
	
	/**广度优先搜索算法：
	 * 1.用remove()方法检查栈顶的顶点
	 * 2.试图找到这个顶点还未访问的邻接点
	 * 3.如果没有找到，该顶点出列
	 * 4.如果找到这样的顶点，访问这个顶点，并把它放入队列中
	 */
	public void breadthFirstSearch(){
		vertexList[0].wasVisited = true;
		displayVertex(0);
		queue.insert(0);
		int v2;
		
		while(!queue.isEmpty()){
			int v1=queue.remove();
			while((v2=getAdjUnvisitedVertex(v1))!=-1){
				vertexList[v2].wasVisited = true;
				displayVertex(v2);
				queue.insert(v2);
			}
		}
		
		for(int i=0;i<nVerts;i++){

			vertexList[i].wasVisited=false;
		}
	}
}
```
```java
//最小生成树
/**基于深度优先搜索找到最小生成树
*这里是指，用最少的边连接所有顶点。对于一组顶点，可能有多种最小生成树，但是最小生成树的边的数量E总是比顶点V的数量小1，即：V = E+1
*/

public void mst(){
    vertexList[0].wasVisited = true;
    theStack.push(0);
     
    while(!theStack.isEmpty()){
        int currentVertex = theStack.peek();
        int v = getAdjUnvisitedVertex(currentVertex);
        if(v == -1){
            theStack.pop();
        }else{
            vertexList[v].wasVisited = true;
            theStack.push(v);
             
            displayVertex(currentVertex);
            displayVertex(v);
            System.out.print(" ");
        }
    }
     
    //搜索完毕，初始化，以便于下次搜索
    for(int i = 0; i < nVerts; i++) {
        vertexList[i].wasVisited = false;
    }
}
```
#### 6.2实现图的深度优先搜索、广度优先搜索
```java
//这块参考6.1中代码
```
#### 6.3.1 实现DijKstra算法
DijKstra算法思想：
互补松弛条件：
设标量d1,d2,...,dN满足
	dj<=di+aij,(i,j)属于A，
且P是以i1为起点ik为终点的路，如果，
	dj=di+aij，对P的所有边（i,j）
成立，那么P是从i到ik的最短路。其中，满足上面两式的被称为最短路问题松弛条件。

1，令G=（V,E）为一个带权无向图。G中若有两个相邻的节点，i，j。aij为节点i到j的权值，在本算法可以理解为距离。每个节点斗鱼一个值di(节点标记)表示其从起点到它的某条路的距离
2，算法储是有个数组V用于存储未访问的节点列表，我们称为候选列表。选定节点1为起始节点。开始时，节点1的d1=0,其他节点di=无穷大，V为所有节点。初始化条件后，然后开始迭代算法，指导B为空集时停止。具体迭代步骤如下：
	将d值最小的节点di从候选列表中移除。（本例中V的数据结构采用的是优先队列实现最小值出列，最好使用斐波那契数列？）对于该节点为期待你的每一条边，不包括移除V的节点，(i,j)属于A，若dj>di+dj（违反松弛条件）

```java
/**DijKstra算法
*这里使用带权无向图，弥补6.1中知识
*/
public class Vertex implements Comparable<Vertex>{

    /**
     * 节点名称(A,B,C,D)
     */
    private String name;
    
    /**
     * 最短路径长度
     */
    private int path;
    
    /**
     * 节点是否已经出列(是否已经处理完毕)
     */
    private boolean isMarked;
    
    public Vertex(String name){
        this.name = name;
        this.path = Integer.MAX_VALUE; //初始设置为无穷大
        this.setMarked(false);
    }
    
    public Vertex(String name, int path){
        this.name = name;
        this.path = path;
        this.setMarked(false);
    }
    
    @Override
    public int compareTo(Vertex o) {
        return o.path > path?-1:1;
    }
}
public class Vertex implements Comparable<Vertex>{

    /**
     * 节点名称(A,B,C,D)
     */
    private String name;
    
    /**
     * 最短路径长度
     */
    private int path;
    
    /**
     * 节点是否已经出列(是否已经处理完毕)
     */
    private boolean isMarked;
    
    public Vertex(String name){
        this.name = name;
        this.path = Integer.MAX_VALUE; //初始设置为无穷大
        this.setMarked(false);
    }
    
    public Vertex(String name, int path){
        this.name = name;
        this.path = path;
        this.setMarked(false);
    }
    
    @Override
    public int compareTo(Vertex o) {
        return o.path > path?-1:1;
    }
}

public class Graph {

    /*
     * 顶点
     */
    private List<Vertex> vertexs;

    /*
     * 边
     */
    private int[][] edges;

    /*
     * 没有访问的顶点
     */
    private Queue<Vertex> unVisited;

    public Graph(List<Vertex> vertexs, int[][] edges) {
        this.vertexs = vertexs;
        this.edges = edges;
        initUnVisited();
    }
    
    /*
     * 搜索各顶点最短路径
     */
    public void search(){
        while(!unVisited.isEmpty()){
            Vertex vertex = unVisited.element();
            //顶点已经计算出最短路径，设置为"已访问"
          　　vertex.setMarked(true);    
            //获取所有"未访问"的邻居
            　　List<Vertex> neighbors = getNeighbors(vertex);    
            //更新邻居的最短路径
            updatesDistance(vertex, neighbors);        
            pop();
        }
        System.out.println("search over");
    }
    
    /*
     * 更新所有邻居的最短路径
     */
    private void updatesDistance(Vertex vertex, List<Vertex> neighbors){
        for(Vertex neighbor: neighbors){
            updateDistance(vertex, neighbor);
        }
    }
    
    /*
     * 更新邻居的最短路径
     */
    private void updateDistance(Vertex vertex, Vertex neighbor){
        int distance = getDistance(vertex, neighbor) + vertex.getPath();
        if(distance < neighbor.getPath()){
            neighbor.setPath(distance);
        }
    }

    /*
     * 初始化未访问顶点集合
     */
    private void initUnVisited() {
        unVisited = new PriorityQueue<Vertex>();
        for (Vertex v : vertexs) {
            unVisited.add(v);
        }
    }

    /*
     * 从未访问顶点集合中删除已找到最短路径的节点
     */
    private void pop() {
        unVisited.poll();
    }

    /*
     * 获取顶点到目标顶点的距离
     */
    private int getDistance(Vertex source, Vertex destination) {
        int sourceIndex = vertexs.indexOf(source);
        int destIndex = vertexs.indexOf(destination);
        return edges[sourceIndex][destIndex];
    }

    /*
     * 获取顶点所有(未访问的)邻居
     */
    private List<Vertex> getNeighbors(Vertex v) {
        List<Vertex> neighbors = new ArrayList<Vertex>();
        int position = vertexs.indexOf(v);
        Vertex neighbor = null;
        int distance;
        for (int i = 0; i < vertexs.size(); i++) {
            if (i == position) {
                //顶点本身，跳过
                continue;
            }
            distance = edges[position][i];    //到所有顶点的距离
            if (distance < Integer.MAX_VALUE) {
                //是邻居(有路径可达)
                neighbor = getVertex(i);
                if (!neighbor.isMarked()) {
                    //如果邻居没有访问过，则加入list;
                    neighbors.add(neighbor);
                }
            }
        }
        return neighbors;
    }

    /*
     * 根据顶点位置获取顶点
     */
    private Vertex getVertex(int index) {
        return vertexs.get(index);
    }

    /*
     * 打印图
     */
    public void printGraph() {
        int verNums = vertexs.size();
        for (int row = 0; row < verNums; row++) {
            for (int col = 0; col < verNums; col++) {
                if(Integer.MAX_VALUE == edges[row][col]){
                    System.out.print("X");
                    System.out.print(" ");
                    continue;
                }
                System.out.print(edges[row][col]);
                System.out.print(" ");
            }
            System.out.println();
        }
    }
}

```
#### 6.3.2 A* 算法
```java
public class AstarPathFind {
   // 前四个是上下左右，后四个是斜角
   public final static int[] dx = { 0, -1, 0, 1, -1, -1, 1, 1 };
   public final static int[] dy = { -1, 0, 1, 0, 1, -1, -1, 1 };
 
   // 最外圈都是1表示不可通过
   final static public int[][] map = {
       { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
       { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
 
   public static void main(String[] args) {
     // TODO Auto-generated method stub
     Point start = new Point(1, 1);
     Point end = new Point(10, 13);
     /*
      * 第一个问题：起点FGH需要初始化吗？
      * 看参考资料的图片发现不需要
      */
     Stack<Point> stack = printPath(start, end);
     if(null==stack) {
       System.out.println("不可达");
     }else {
       while(!stack.isEmpty()) {
         //输出(1,2)这样的形势需要重写toString
         System.out.print(stack.pop()+" -> ");
       }
       System.out.println();
     }
 
   }
 
   public static Stack<Point> printPath(Point start, Point end) {
     
     /*
      * 不用PriorityQueue是因为必须取出存在的元素
      */
     ArrayList<Point> openTable = new ArrayList<Point>();
     ArrayList<Point> closeTable = new ArrayList<Point>();
     openTable .clear();
     closeTable.clear();
     Stack<Point> pathStack = new Stack<Point>();
     start.parent = null;
     //该点起到转换作用，就是当前扩展点
     Point currentPoint = new Point(start.x, start.y);
     //closeTable.add(currentPoint);
     boolean flag = true;
     
     while(flag) {
       for (int i = 0; i < 8; i++) {
         int fx = currentPoint.x + dx[i];
         int fy = currentPoint.y + dy[i];
         Point tempPoint = new Point(fx,fy);
         if (map[fx][fy] == 1) {
           // 由于边界都是1中间障碍物也是1，，这样不必考虑越界和障碍点扩展问题
           //如果不设置边界那么fx >=map.length &&fy>=map[0].length判断越界问题
           continue;
         } else {
           if(end.equals(tempPoint)) {
             flag = false;
             //不是tempPoint，他俩都一样了此时
             end.parent = currentPoint;
             break;
           }
           if(i<4) {
             tempPoint.G = currentPoint.G + 10;
           }else {
             tempPoint.G = currentPoint.G + 14;
           }
           tempPoint.H = Point.getDis(tempPoint,end);
           tempPoint.F = tempPoint.G + tempPoint.H;
           //因为重写了equals方法，所以这里包含只是按equals相等包含
           //这一点是使用java封装好类的关键
           if(openTable.contains(tempPoint)) {
             int pos = openTable.indexOf(tempPoint );
             Point temp = openTable.get(pos);
             if(temp.F > tempPoint.F) {
               openTable.remove(pos);
               openTable.add(tempPoint);
               tempPoint.parent = currentPoint;
             }
           }else if(closeTable.contains(tempPoint)){
             int pos = closeTable.indexOf(tempPoint );
             Point temp = closeTable.get(pos);
             if(temp.F > tempPoint.F) {
               closeTable.remove(pos);
               openTable.add(tempPoint);
               tempPoint.parent = currentPoint;
             }
           }else {
             openTable.add(tempPoint);
             tempPoint.parent = currentPoint;
           }
 
         }
       }//end for
       
       if(openTable.isEmpty()) {
         return null;
       }//无路径
       if(false==flag) {
         break;
       }//找到路径
       openTable.remove(currentPoint);
       closeTable.add(currentPoint);
       Collections.sort(openTable);
       currentPoint = openTable.get(0);
       
     }//end while
     Point node = end;
     while(node.parent!=null) {
       pathStack.push(node);
       node = node.parent;
     }    
     return pathStack;
   }
 }
 
 class Point implements Comparable<Point>{
   int x;
   int y;
   Point parent;
   int F, G, H;
 
   public Point(int x, int y) {
     super();
     this.x = x;
     this.y = y;
     this.F = 0;
     this.G = 0;
     this.H = 0;
   }
 
   @Override
   public int compareTo(Point o) {
     // TODO Auto-generated method stub
     return this.F  - o.F;
   }
 
   @Override
   public boolean equals(Object obj) {
     Point point = (Point) obj;
     if (point.x == this.x && point.y == this.y)
       return true;
     return false;
   }
 
   public static int getDis(Point p1, Point p2) {
     int dis = Math.abs(p1.x - p2.x) * 10 + Math.abs(p1.y - p2.y) * 10;
     return dis;
   }
 
   @Override
   public String toString() {
     return "(" + this.x + "," + this.y + ")";
   } 
 }
```


#### 6.4实现拓扑排序的 Kahn 算法、DFS 算法
拓扑排序是对有向无圈图的顶点的一种排序，使得如果存在一条从vi到vj的路径，那么在拓扑排序中，vj就出现在vi的后面。
拓扑图中，不能出现圈，如果有圈，那么就没有意义。
一个有向图能被拓扑排序的充要条件就是它是一个有向无环图。
偏序：在有向图中两个顶点之间不存在环路，至于连通与否，是无所谓的。所以，有向无环图必然是满足偏序关系的。
全序：在偏序的基础之上，有向无环图中的任意一对顶点还需要有明确的关系，反应在图中，就是单向连通的关系。（不能双向连通，否则就是环）
```java
// Kahn 算法
public class KahnTopological  
{  
    private List<Integer> result;   // 用来存储结果集  
    private Queue<Integer> setOfZeroIndegree;  // 用来存储入度为0的顶点  
    private int[] indegrees;  // 记录每个顶点当前的入度  
    private int edges;  
    private Digraph di;  
      
    public KahnTopological(Digraph di)  
    {  
        this.di = di;  
        this.edges = di.getE();  
        this.indegrees = new int[di.getV()];  
        this.result = new ArrayList<Integer>();  
        this.setOfZeroIndegree = new LinkedList<Integer>();  
          
        // 对入度为0的集合进行初始化  
        Iterable<Integer>[] adjs = di.getAdj();  
        for(int i = 0; i < adjs.length; i++)  
        {  
            // 对每一条边 v -> w   
            for(int w : adjs[i])  
            {  
                indegrees[w]++;  
            }  
        }  
          
        for(int i = 0; i < indegrees.length; i++)  
        {  
            if(0 == indegrees[i])  
            {  
                setOfZeroIndegree.enqueue(i);  
            }  
        }  
        process();  
    }  
      
    private void process()  
    {  
        while(!setOfZeroIndegree.isEmpty())  
        {  
            int v = setOfZeroIndegree.dequeue();  
              
            // 将当前顶点添加到结果集中  
            result.add(v);  
              
            // 遍历由v引出的所有边  
            for(int w : di.adj(v))  
            {  
                // 将该边从图中移除，通过减少边的数量来表示  
                edges--;  
                if(0 == --indegrees[w])   // 如果入度为0，那么加入入度为0的集合  
                {  
                    setOfZeroIndegree.enqueue(w);  
                }  
            }  
        }  
        // 如果此时图中还存在边，那么说明图中含有环路  
        if(0 != edges)  
        {  
            throw new IllegalArgumentException("Has Cycle !");  
        }  
    }  
      
    public Iterable<Integer> getResult()  
    {  
        return result;  
    }  
}  

//DFS算法
public class DirectedDepthFirstOrder  
{  
    // visited数组，DFS实现需要用到  
    private boolean[] visited;  
    // 使用栈来保存最后的结果  
    private Stack<Integer> reversePost;  
  
    /** 
     * Topological Sorting Constructor 
     */  
    public DirectedDepthFirstOrder(Digraph di, boolean detectCycle)  
    {  
        // 这里的DirectedDepthFirstCycleDetection是一个用于检测有向图中是否存在环路的类  
        DirectedDepthFirstCycleDetection detect = new DirectedDepthFirstCycleDetection(  
                di);  
          
        if (detectCycle && detect.hasCycle())  
            throw new IllegalArgumentException("Has cycle");  
              
        this.visited = new boolean[di.getV()];  
        this.reversePost = new Stack<Integer>();  
  
        for (int i = 0; i < di.getV(); i++)  
        {  
            if (!visited[i])  
            {  
                dfs(di, i);  
            }  
        }  
    }  
  
    private void dfs(Digraph di, int v)  
    {  
        visited[v] = true;  
  
        for (int w : di.adj(v))  
        {  
            if (!visited[w])  
            {  
                dfs(di, w);  
            }  
        }  
  
        // 在即将退出dfs方法的时候，将当前顶点添加到结果集中  
        reversePost.push(v);  
    }  
  
    public Iterable<Integer> getReversePost()  
    {  
        return reversePost;  
    }  
}  
```
#### 6.5Number of Islands（岛屿的个数）
```java
class Solution {	
    public int numIslands(char[][] grid) {
        int count=0;
        for(int x=0;x<grid.length;x++)
        	for(int y =0;y<grid[0].length;y++) {
        		if( grid[x][y] =='1') {
        			clear(grid,x,y);
        			++count;
        		}
        	}
		return count;
    	
    }   
    public void clear(char[][] grid,int x,int y) {
    	grid[x][y]=0;
    	if(x+1<grid.length&& grid[x+1][y]=='1')clear(grid,x+1,y);
    	if(x-1>=0&& grid[x-1][y]=='1')clear(grid,x-1,y);
    	if(y+1<grid[0].length&& grid[x][y+1]=='1')clear(grid,x,y+1);
    	if(y-1>=0&& grid[x][y-1]=='1')clear(grid,x,y-1);	
    }
}
```

#### 6.6Valid Sudoku（有效的数独）
```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        boolean[] rowFlags = new boolean[9];
        boolean[] colFlags = new boolean[9];
        boolean[] squareFlags = new boolean[9];

        for (int i = 0; i < 9; i++) {
            Arrays.fill(rowFlags, false);
            Arrays.fill(colFlags, false);
            Arrays.fill(squareFlags, false);
            for (int j = 0; j < 9; j++) {
                // 行数独
                char cell = board[i][j];
                if (!isCellValid(rowFlags, cell)) {
                    return false;
                }
                // 列数独
                cell = board[j][i];
                if (!isCellValid(colFlags, cell)) {
                    return false;
                }
                // 3*3 方格数独
                int row = (j / 3) + ((i / 3) * 3);
                int col = (j % 3) + ((i % 3) * 3);
                cell = board[row][col];
                if (!isCellValid(squareFlags, cell)) {
                    return false;
                }
            }
        }

        return true;
    }
	//如果之前出现过，就return false。
    public boolean isCellValid(boolean[] flags, char cell) {
        if (cell == '.') {
            return true;
        }
        int value = cell - 49;
        if (flags[value]) {
            return false;
        }
        flags[value] = true;
        return true;
    }
}
```


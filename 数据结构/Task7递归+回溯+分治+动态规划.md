# Datawhale 系列数据结构
本文参考链接：
01背包问题：https://blog.csdn.net/chanmufeng/article/details/82955730 

## Task7.1  递归
#### 7.1.1爬楼梯
```java
//爬楼梯：
//假设你正在爬楼梯。需要 n 阶你才能到达楼顶
//每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
class Solution {
    public int climbStairs(int n) {
        int [] ways = new int[n+1];
        ways[0] = 0;
        for (int i = 1;i<ways.length;i++){
            if (i < 3 ){
                ways[i] = i;
            }else {
                ways[i] = ways[i-1] + ways[i-2];
            }
        }
        return ways[n];
    }
}
//使用最小花费爬楼梯
//数组的每个索引做为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
//每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
//您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。
class Solution {
    public int minCostClimbingStairs(int[] cost) {
       int length = cost.length;
        int[] newCost = new int[length+1];
        newCost = Arrays.copyOf(cost,length+1);
        length = length+1;
        cost = newCost;
        int[] fn = new int[length];
        fn[0] = 0;
        fn[1] = 0;
        fn[2] = Math.min(cost[0],cost[1]);
        for (int i = 3;i<length;i++){
            fn[i] = Math.min(fn[i-1] + cost[i-1],fn[i-2]+cost[i-2]);
        }
        return fn[length-1]; 
    }
}
```
#### 7.1.2 0-1背包问题（递归方法解决）
问题描述：
给定 n 种物品重量$w_1,w_2,w_3,...,w_n$,价值为$v_1,v_2,v_3,...,v_4$，一个容量为$C$的背包，问：应该如何选择装入背包的物品，使得装入背包中的物品的总价值最大？
递归思想：
首先我们用递归的方式来尝试解决这个问题
我们用$F(n,C)$ 表示将前$n$个物品放进容量为$C$的背包里，得到最大的价值。
我们用自顶向下的角度来看，假如我们已经进行到了最后一步（即求解将$n$个物品放到背包了获得最大价值），此时我们便有两种选择：
	1.不放第$n$个物品，此时总价值为$F(n-1,C)$
	2.放置第$n$个物品，此时总价值为$v_n+F(n-1,C-w_n)$
两种选择中总价值最大的方案就是我们的最终方案，递推式如下：
	$F(i,C)=max(F(i-1,C),v(i)+F(i-1,C-w(i)))$
```java
//递归解法
public static int solveKS(int[] w,int[] v,int index,int capacity ){
		if(index<0 || capacity <=0) return 0;
		
		int res = solveKS(w,v,index-1,capacity);
		if(w[index]<=capacity){
			res=Math.max(res, v[index]+solveKS(w,v,index-1,capacity-w[index]));
		}
		return res;
	}
	
	public static int knapSack(int [] w,int [] v,int C){
		int size=w.length;
		return solveKS(w,v,size-1,C);
	}
```
## Task 7.2 回溯
#### 7.2.1八皇后问题
```java
public static int [][] array=new int[8][8];
	public static int map=0;
	
	public static void main(String[] args) {
		System.out.println("八皇后问题");
	    findQueen(0);
	    System.out.println("八皇后问题共有："+map+"种可能");
	}
	
	public static void findQueen(int i){
		if(i>7){
			map++;
			print();//
			return;
		}
		for(int m=0;m<8;m++){
			if(check(i,m)){
				array[i][m]=1;
				findQueen(i+1);
				array[i][m]=0;
			}
		}
	}
	public static boolean check(int k, int j){
		for(int i=0;i<8;i++){//检查行列冲突
			if(array[i][j]==1){
				return false;
			}
		}
		for(int i=k-1,m=j-1;i>=0&& m>=0;i--,m--){
			if(array[i][m]==1){//检查左对角线冲突
				return false;
			}
		}
		for(int i=k-1,m=j+1;i>=0&&m<=7;i--,m++){
			if(array[i][m]==1){
				return false;
			}
		}
		return true;
	}
	public static void print(){
		System.out.print("方案"+map+":"+"\n");
	    for(int i=0;i<8;i++){
	        for(int m=0;m<8;m++){
	            if(array[i][m]==1){  
	                //System.out.print("皇后"+(i+1)+"在第"+i+"行，第"+m+"列\t");
	                System.out.print("o ");
	            }
	            else{
	                    System.out.print("+ ");
	            }
	        }
	        System.out.println();
	    }
	    System.out.println();
	}
```
#### 7.2.2求解0-1背包问题（回溯方法解决）
整体思路：
	用回溯法需要构造子集树。对于每一个物品i,对于该物品只有选与不选2个决策，总共有n个物品，可以顺序依次考虑每个物品，这样就形成了一颗空间树：基本思想就是遍历这棵树，以枚举所有情况，最后进行判断，如果重量不超过背包容量，且价值最大的话，该方案就是最后的答案
算法设计：
	利用回溯设计一个算法求出0-1背包问题的解，也就是求出一个解向量xi(即对n个物品放或不妨的一种的方案)
	其中，（xi=0或1，xi=0表示物体i不放入背包，xi=1表示把物体i放入背包）。
	在递归函数Backtrack中，
		当i>n时，算法搜索到叶子节点，得到一个新的物品包装方案。此时算法适时更新当前的最有价值。
		当i<n时，当前扩展结点位于排列树的第（i-1）层，此时算法选择下一个要安排的物品，以深度优先方式递归的对相应的子树进行搜索，对不满足上界约束的结点，则剪去相应的子树
```java
//回溯
public class KnapSack02 {
	private static int n=3;//物品数量编号，从0开始
	private static double c=5;//背包容量
	private static double [] v={12,10,20,15};//各个物品的价值
	private static double [] w={2,1,3,2};//各个物品的重量
	private static double cw = 0.0;//当前背包重量　current weight
	private static double cp = 0.0;//当前背包中物品总价值　current value
	private static double bestp = 0.0;//当前最优价值best price
	private static double [] perp = new double [4];//单位物品价值(排序后) per price
	private static int [] order = new int [4];//物品编号
	private static int [] put = new int [4];//设置是否装入，1装入，0不装

	//按单位价值排序
	public static  void knapsack(){
		int i,j;
		int temporder = 0;
		double temp= 0.0;
		
		for(i=1;i<=n;i++)
			perp[i]=v[i]/w[i];
		  for(i=1;i<=n-1;i++)
		    {
		        for(j=i+1;j<=n;j++)
		            if(perp[i]<perp[j])//冒泡排序perp[],order[],sortv[],sortw[]
		        {
		            temp = perp[i];  //冒泡对perp[]排序
		            perp[i]=perp[i];
		            perp[j]=temp;
		 
		            temporder=order[i];//冒泡对order[]排序
		            order[i]=order[j];
		            order[j]=temporder;
		 
		            temp = v[i];//冒泡对v[]排序
		            v[i]=v[j];
		            v[j]=temp;
		 
		            temp=w[i];//冒泡对w[]排序
		            w[i]=w[j];
		            w[j]=temp;
		        }
		    }
	}
	
	public static void backtrack(int i){
		//i表示到达的层数（第几步，从0开始），同事也只是当前选择玩了几个物品 
		bound( i);
		if(i>n){
			bestp=cp;
			return;
		}
		 //如若左子节点可行，则直接搜索左子树;
	    //对于右子树，先计算上界函数，以判断是否将其减去
	    if(cw+w[i]<=c)//将物品i放入背包,搜索左子树
	    {
	        cw+=w[i];//同步更新当前背包的重量
	        cp+=v[i];//同步更新当前背包的总价值
	        put[i]=1;
	        backtrack(i+1);//深度搜索进入下一层
	        cw-=w[i];//回溯复原
	        cp-=v[i];//回溯复原
	    }
	    if(bound(i+1)>bestp)//如若符合条件则搜索右子树
	        backtrack(i+1);
	}
	
	//计算上界函数，功能为剪枝
	public static double bound(int i)
	{   //判断当前背包的总价值cp＋剩余容量可容纳的最大价值<=当前最优价值
	    double leftw= c-cw;//剩余背包容量
	    double b = cp;//记录当前背包的总价值cp,最后求上界
	    //以物品单位重量价值递减次序装入物品
	    while(i<=n && w[i]<=leftw)
	    {
	        leftw-=w[i];
	        b+=v[i];
	        i++;
	    }
	    //装满背包
	    if(i<=n)
	        b+=v[i]/w[i]*leftw;
	    return b;//返回计算出的上界
	 
	}
	
	public static void main(String[] args) {
	     knapsack();
	     backtrack(1);
	     System.out.println(bestp);
	}
	
}
```
## Task7.3 分治
#### 7.3.1 利用分治算法求一组数据的逆序对个数
```java
public class ReverseOrder {
	private static int sum = 0;
	private static int []a ={5,4,2,6,3,1};
	private static int []b =new int [6]; 
	
	public static void worksort(int l,int r){
		int mid,tmp,i,j;
		if(r>l+1){
			mid=(l+r)/2;
			worksort(l,mid-1);
			worksort(mid,r);
			tmp=l;
			for(i=l,j=mid;i<=mid-1 && j<=r;){
				if(a[i]>a[j])
				{
					b[tmp++]=a[j++];//快速排序 
					sum+=mid-i;//统计逆序对个数 
				}
				else
				   b[tmp++]=a[i++];
			}
			 if(j<=r)
				  for(;j<=r;j++)  b[tmp++]=a[j];
			 else
				  for(;i<=mid-1;i++)   b[tmp++]=a[i];
			 for(i=l;i<=r;i++)  a[i]=b[i];//将排好序的b数组赋值给a数组
		}else{
			if(l+1==r)//递归的边界 
				  if(a[l]>a[r])
				  { int temp = a[l];
				  	a[l]=a[r];
				  	a[r]=temp;
				  	sum++;
				  }
		}
	}
	public static void main(String[] args) {
		worksort(0,5);
		System.out.println(sum);
	}
}
```

## Task 7.4 动态规划
#### 7.4.1 0-1背包问题
```java
//动态规划
int size = w.length;
        if (size == 0) {
            return 0;
        }

        int[] dp = new int[C + 1];
        //初始化第一行
        //仅考虑容量为C的背包放第0个物品的情况
        for (int i = 0; i <= C; i++) {
            dp[i] = w[0] <= i ? v[0] : 0;
        }

        for (int i = 1; i < size; i++) {
            for (int j = C; j >= w[i]; j--) {
                dp[j] = Math.max(dp[j], v[i] + dp[j - w[i]]);
            }
        }
        return dp[C];
    }

    public static void main(String[] args) {
        int[] w = {2, 1, 3, 2};
        int[] v = {12, 10, 20, 15};
        System.out.println(knapSack(w, v, 5));
    }
```
#### 7.4.2 编程实现莱温斯坦最短编辑距离
```java
public static int minEditDistance(String dest,String src){
		int [][] f=new int[dest.length()+1][src.length()+1];
		f[0][0]=0;
		for(int i=1;i<dest.length()+1;i++){
			f[i][0]=i;
		}
		for(int i=1;i<src.length()+1;i++){
			f[0][i]=i;
		}
		for(int i=1;i<dest.length()+1;i++){
			for(int j=1;j<src.length()+1;j++){
				int cost=0;
				if(dest.charAt(i - 1) != src.charAt(j - 1)){
					cost=1;
				}
				int minCost;
				if(f[i-1][j]<f[i][j-1]){
					minCost=f[i-1][j]+cost;
				}else{
					minCost=f[i][j-1]+cost;
				}
				f[i][j]=minCost;
			}	
		}
		return f[dest.length()][src.length()];
	}
	public static void main(String[] args) {
		
		System.out.println(minEditDistance("sot", "stop"));
	}
```

#### 7.4.3 编程实现查找两个字符串的最长子序列
```java
//求解str1 和 str2 的最长公共子序列
    public static int LCS(String str1, String str2){
        int[][] c = new int[str1.length() + 1][str2.length() + 1];
        for(int row = 0; row <= str1.length(); row++)
            c[row][0] = 0;
        for(int column = 0; column <= str2.length(); column++)
            c[0][column] = 0;
        
        for(int i = 1; i <= str1.length(); i++)
            for(int j = 1; j <= str2.length(); j++)
            {
                if(str1.charAt(i-1) == str2.charAt(j-1))
                    c[i][j] = c[i-1][j-1] + 1;
                else if(c[i][j-1] > c[i-1][j])
                    c[i][j] = c[i][j-1];
                else
                    c[i][j] = c[i-1][j];
            }
        return c[str1.length()][str2.length()];
    }
    
    //test
    public static void main(String[] args) {
        String str1 = "BDCABA";
        String str2 = "ABCBDAB";
        int result = LCS(str1, str2);
        System.out.println(result);
    }
```
#### 7.4.4编程实现一个数据序列的最长递增子序列
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        /**
        dp[i]: 所有长度为i+1的递增子序列中, 最小的那个序列尾数.
        由定义知dp数组必然是一个递增数组, 可以用 maxL 来表示最长递增子序列的长度. 
        对数组进行迭代, 依次判断每个数num将其插入dp数组相应的位置:
        1. num > dp[maxL], 表示num比所有已知递增序列的尾数都大, 将num添加入dp
           数组尾部, 并将最长递增序列长度maxL加1
        2. dp[i-1] < num <= dp[i], 只更新相应的dp[i]
        **/
        int maxL = 0;
        int[] dp = new int[nums.length];
        for(int num : nums) {
            // 二分法查找, 也可以调用库函数如binary_search
            int lo = 0, hi = maxL;
            while(lo < hi) {
                int mid = lo+(hi-lo)/2;
                if(dp[mid] < num)
                    lo = mid+1;
                else
                    hi = mid;
            }
            dp[lo] = num;
            if(lo == maxL)
                maxL++;
        }
        return maxL;
    }
}
```
## Task 7.5 练习
#### 7.5.1 实战递归
```java
//Letter Combinations of a Phone Number(17)
String[] button=new String[]{"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};

    List<String> list=new ArrayList<>();
    public List<String> letterCombinations(String digits) {
        if (digits==null||digits.length()==0)
            return list;
        letterCombinations(digits,0,new String());
        return list;
    }
    public void  letterCombinations(String digits,int index,String temp) {
       if(index==digits.length()){
            list.add(temp);
            return;
        }
        int position=digits.charAt(index)-'0';
        String str=button[position];
        for (int i=0;i<str.length();i++){
            letterCombinations(digits,index+1,temp+str.charAt(i));
        }
    }

//permutations(46)
 List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        permutation(nums, 0, nums.length - 1);
        return res;
    }
    
    private void permutation(int[] nums, int p, int q) {
        if (p == q) {
            res.add(arrayToList(nums));
        }
        for (int i = p; i <= q; i++) {
            swap(nums, i, p);
            permutation(nums, p + 1, q);
            swap(nums, i, p);  // 这里要交换回来，免得出现重复的情况
        }
    }
    
    private List<Integer> arrayToList(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            res.add(nums[i]);
        }
        return res;
    }
    
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
```
#### 7.5.2 实战dp
```java
//完成0-1背包问题实现(自我实现)及Leetcode
参考7.4.1
//Palindrome Partitioning II(132)
 public int minCut(String s) {
        if(s == null || s.isEmpty()){
            return 0;
        }
        int len = s.length();
        boolean[][] dp = new boolean[s.length()][s.length()];
        int[] cut = new int[s.length()];

        for(int i = 0; i < len; i++){
            //最大划分就是i次
            cut[i]= i;
            for(int j = 0; j <= i; j++){
                if(s.charAt(i) == s.charAt(j) &&(i-j <= 1 || dp[j+1][i-1])){
                    dp[j][i] = true;
                    if(j == 0) {
                        //0-i直接是回文
                        cut[i] = 0;
                    } else {
                        cut[i] = Math.min(cut[j-1]+1, cut[i]);
                    }
                }
            }
        }
        return cut[len-1];
    }

```

#### 7.5.2 可选练习
```java
//Regular Expression Matching（正则表达式匹配）
class Solution {
    public boolean isMatch(String text, String pattern) {
        if (pattern.isEmpty()) return text.isEmpty();
        boolean first_match = (!text.isEmpty() &&
                               (pattern.charAt(0) == text.charAt(0) || pattern.charAt(0) == '.'));

        if (pattern.length() >= 2 && pattern.charAt(1) == '*'){
            return (isMatch(text, pattern.substring(2)) ||
                    (first_match && isMatch(text.substring(1), pattern)));
        } else {
            return first_match && isMatch(text.substring(1), pattern.substring(1));
        }
    }
}
//Minimum Path Sum（最小路径和）
public int minPathSum(int[][] grid) {
        int rows=grid.length;
        int cols=grid[0].length;
        
        int [] dp=new int[cols];
        dp[0]=grid[0][0];
        
        for(int col=1;col<cols;col++){
            dp[col]=dp[col-1]+grid[0][col];
        }
        for(int row = 1; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                if(col > 0){
                    dp[col] = Math.min(dp[col-1], dp[col]) + grid[row][col];
                }else{
                    dp[col] += grid[row][col];
                }
            }
        }
         return dp[cols - 1];
    }
//Coin Change （零钱兑换）[作为可选]
//Best Time to Buy and Sell Stock（买卖股票的最佳时机）[作为可选]
//Maximum Product Subarray（乘积最大子序列）[作为可选]
//Triangle（三角形最小路径和）[作为可选]
```


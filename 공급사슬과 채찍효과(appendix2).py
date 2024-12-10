from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from datetime import datetime

# 현재 시간 가져오기
current_time = datetime.now()

# 포맷에 맞춰 출력
print("현재 시간:", current_time.strftime("%H:%M:%S"))
save_csv = False
save_fig = False
time = 60
## SCIP 솔버 초기화
solver = pywraplp.Solver.CreateSolver('SCIP')
#%%
######셋 및 상수######
#Products
P = ['Product1']
#central warehouse
W0 = ['Warehouse0']

#regional warehouses
#W = ['Warehouse1', 'Warehouse2']
W = ['Warehouse1','Warehouse2']
#retailers
#R = ['Retailer1', 'Retailer2']
R = ['Retailer1','Retailer2','Retailer3','Retailer4']



#제품 수
NP = len(P)
#지역 창고수
NW = len(W)
#리테일러 수
NR = len(R)
#number of time periods in the cycle demand
T = 7

#time periods
D = [i for i in range(1,T+1)]

#supply chain nodes
#len(I) = NW+NR+1
I = [i for i in W0+W+R]

#demand nodes (regional warehouses and retailers)
#len(DN) = NW+NR
DN = [i for i in W+R]

#supply nodes
#len(SN) = NW+1
SN = [i for i in W0+W]

#######파라미터(입력변수)######
#a large positive number
BGM = 100000000000000
#batch size
BatSiz = 5

#기간 t의 엔티티 J에 있는 제품 I에 대한 고객 수요(고객 수요는 소매업체에서만 발생하고 창고에서는 발생하지 않음)
#CD[i][j][t]
CD={}
R_1 = [7, 8, 11, 10, 20, 36, 38, 6, 7, 10, 12, 20, 36, 39, 7, 8, 11, 13, 19, 35, 37, 5, 9, 10, 11, 22, 34, 39]
R_2 = [6, 9, 10, 16, 18, 35, 36, 7, 8, 11, 14, 16, 36, 38, 6, 9, 12, 13, 20, 34, 36, 8, 9, 12, 15, 15, 34, 37]
R_3 = [6, 8, 10, 14, 16, 37, 39, 5, 6, 9, 16, 18, 37, 39, 6, 8, 9, 17, 19, 35, 36, 6, 7, 9, 16, 19, 36, 37]
R_4 = [7, 8, 9, 13, 21, 34, 38, 8, 9, 10, 13, 19, 33, 38, 5, 7, 8, 12, 24, 36, 38, 8, 9, 11, 12, 21, 34, 35]
#R_1 = [i+j for i,j in zip(R_1,R_2)]
R_D = {R: i for R,i in zip(R,[R_1,R_2,R_3,R_4])}
for i in P:
    CD[i] ={}
    for j in R:
        CD[i][j] ={}
        for t in D:
            CD[i][j][t] = R_D[j][t-1]

#계획기간
H = 56




#기간당 엔티티 J에서 제품 I의 단일 보유 비용
#HOC[i][j]
HOC={}
for i in P:
    HOC[i] = {}
    for j in DN:
        if j in W:
            HOC[i][j] = 0.2
        elif j in R:
            HOC[i][j] = 0.6


#엔티티 J에서 엔티티 K로의 제품 I의 단일 운송 중 보유 비용(운송 중 보유는 배송 작업 및 환적 작업을 위한 것입니다.)
#HTC[i][j][k]
HTC={}
for i in P:
    HTC[i] = {}
    for j in I:
        HTC[i][j]={}
        for k in I:
            if j in W0 and k in W:
                HTC[i][j][k] = 0.3
            elif j in W and k in W and j!=k:
                HTC[i][j][k] = 0.3
            elif j in W and k in R:
                HTC[i][j][k] = 0.9
            elif j in R and k in R and j !=k:
                HTC[i][j][k] = 0.9
            else:
                HTC[i][j][k] = 1000

    
#기간 t에서 엔티티 J에 있는 제품 i의 단일 손실 판매 비용
#LSC[i][k][t]
LSC = {}
for i in P:
    LSC[i] = {}
    for k in DN:
        LSC[i][k]={}
        for t in D:
            if k in R:
                LSC[i][k][t] = 25
            else:
                LSC[i][k][t] = 0

#엔티티 J에서 엔티티 K로의 운송 리드 타임
#LTT[j][k]
LTT = {}

for j in I:
    LTT[j] ={}
    for k in I:
        if j in W0 and k in W:
            LTT[j][k] = 1
        elif j in W and k in W and j != k:
            LTT[j][k] = 1
        elif j in W and k in R:
            if j == 'Warehouse1' and k in 'Retailer1':
                LTT[j][k] = 1
            elif j == 'Warehouse1' and k in 'Retailer2':
                LTT[j][k] = 1
            elif j == 'Warehouse1' and k in 'Retailer3':
                LTT[j][k] = 1
            elif j == 'Warehouse1' and k in 'Retailer4':
                LTT[j][k] = 1
                
            elif j == 'Warehouse2' and k in 'Retailer1':
                LTT[j][k] = 2
            elif j == 'Warehouse2' and k in 'Retailer2':
                LTT[j][k] = 2
            elif j == 'Warehouse2' and k in 'Retailer3':
                LTT[j][k] = 1
            elif j == 'Warehouse2' and k in 'Retailer4':
                LTT[j][k] = 0
            else:
                LTT[j][k] = 100
        elif j in R and k in R and j != k:
            LTT[j][k] = 1
        else:
            #LTT[j][k] = BGM
            LTT[j][k] = 100
        
#엔티티 J에서 제품 I의 주문 비용(주문 비용은 제품 i의 수량과 무관함)
#OC[i][j]    
OC = {}
for i in P:
    OC[i] ={}
    for j in DN:
        if BatSiz == 5:
            OC[i][j] = 10
        elif BatSiz == 10:
            OC[i][j] = 5
        else:
            OC[i][j] = 20

#엔티티 J에 있는 제품 I의 안전 재고
#SS[i][j]
SS = {}
for i in P:
    SS[i] = {}
    for j in I:
        SS[i][j] = 0
        
        
#기간 t의 엔티티 J에 있는 스토리지 용량
#STC[j][t]
STC = {}
for j in I:
    STC[j] = {}
    for t in D:
        if j in W:
            STC[j][t] = 50
        elif j in R:
            STC[j][t] = 25
        else:
            STC[j][t] = BGM

#싸이클 타임
T = T

#엔티티 J에서 엔티티 K로의 최대 운송 용량
#TRACMAX[j][k]
TRACMAX = {}
for j in I:
    TRACMAX[j] = {}
    for k in I:
        TRACMAX[j][k] = 40
        
#엔티티 J에서 엔티티 K로의 최소 운송 용량
#TRACMIN[j][k]
TRACMIN = {}
for j in I:
    TRACMIN[j] ={}
    for k in I:
        TRACMIN[j][k] = 5
        
        
        
#엔티티 J에서 엔티티 K로의 제품 I의 단일 운송 비용(운송은 배송 작업 및 환적 작업을 위한 것임)
#TRC[i][j][k]
TRC = {}
for i in P:
    TRC[i] = {}
    for j in I:
        TRC[i][j] = {}
        for k in I:
            if j == 'Warehouse0' and k == 'Warehouse1':
                TRC[i][j][k] = 0.15
            elif j == 'Warehouse0' and k == 'Warehouse2':
                TRC[i][j][k] = 0.52
            elif j in W and k in W and j!=k:
                TRC[i][j][k] = 0.35
        
            elif j == 'Warehouse1' and k == 'Retailer1':
                TRC[i][j][k] = 0.22
            elif j == 'Warehouse1' and k == 'Retailer2':
                TRC[i][j][k] = 0.2
            elif j == 'Warehouse1' and k == 'Retailer3':
                TRC[i][j][k] = 0.32
            elif j == 'Warehouse1' and k == 'Retailer4':
                TRC[i][j][k] = 0.38
                
            elif j == 'Warehouse2' and k == 'Retailer1':
                TRC[i][j][k] = 0.68
            elif j == 'Warehouse2' and k == 'Retailer2':
                TRC[i][j][k] = 0.52
            elif j == 'Warehouse2' and k == 'Retailer3':
                TRC[i][j][k] = 0.34
            elif j == 'Warehouse2' and k == 'Retailer4':
                TRC[i][j][k] = 0.4
            
            
            elif j in W and k in W and j!=k:
                TRC[i][j][k] = 0.35
                
                
            elif j == 'Retailer1' and k == 'Retailer2':
                TRC[i][j][k] = 0.1
            elif j == 'Retailer1' and k == 'Retailer3':
                TRC[i][j][k] = 0.4
            elif j == 'Retailer1' and k == 'Retailer4':
                TRC[i][j][k] = 0.65
            
            
            
            elif j == 'Retailer2' and k == 'Retailer1':
                TRC[i][j][k] = 0.1
            elif j == 'Retailer2' and k == 'Retailer3':
                TRC[i][j][k] = 0.15
            elif j == 'Retailer2' and k == 'Retailer4':
                TRC[i][j][k] = 0.5
                
                
            elif j == 'Retailer3' and k == 'Retailer2':
                TRC[i][j][k] = 0.15
            elif j == 'Retailer3' and k == 'Retailer1':
                TRC[i][j][k] = 0.4
            elif j == 'Retailer3' and k == 'Retailer4':
                TRC[i][j][k] = 0.18

                
            elif j == 'Retailer4' and k == 'Retailer2':
                TRC[i][j][k] = 0.5
            elif j == 'Retailer4' and k == 'Retailer3':
                TRC[i][j][k] = 0.18
            elif j == 'Retailer4' and k == 'Retailer1':
                TRC[i][j][k] = 0.65
                
                
            
            
            else:
                TRC[i][j][k] = BGM

######Non-negative continuous variables#####
#기간 t의 끝 시점에 엔티티 J의 제품 I 재고
#FI[i][j][t]
FI = {i : {j : {t : solver.NumVar(0,BGM, f'FI{i}{j}{t}') for t in D} for j in I} for i in P}
#기간 t의 끝에서 엔티티 J의 제품 판매 수량 손실(판매 손실은 소매점에서만 발생함)
#LS[i][j][t]
LS = {i : {j : {t : solver.NumVar(0,BGM, f'LS{i}{j}{t}') for t in D} for j in I} for i in P}
#기간 t 동안 엔티티 J에서 엔티티 K로 제품 i의 배송 수량
#SQ[i][j][k][t]
SQ = {i : {j : {k :{t : solver.NumVar(0,BGM, f'SQ{i}{j}{k}{t}') for t in D} for k in I} for j in I} for i in P}


########Non-negative integer variable#########
#각 기간 t에서 엔터티 J와 엔터티 K 사이의 곱 i 배치 수
#NumBat[i][j][k][t]
NumBat = {i : {j : {k :{t : solver.IntVar(0, BGM, f'NumBat{i}{j}{t}') for t in D} for k in I} for j in I} for i in P}

###########Binary variable#############
#제품 i의 주문이 기간 t에서 엔티티 j에 의해 발주되는 경우 1과 같습니다. 그렇지 않으면 0
#BV1[i][j][t]
BV1 = {i : {j : {t : solver.IntVar(0, 1, f'BV1_{i}{j}{t}') for t in D} for j in I} for i in P}



#
BV2 = {i : {j : {k :{t : solver.IntVar(0, 1, f'BV2_{i}{j}{k}{t}') for t in D} for k in I} for j in I} for i in P}


########Time operator##########
# wrap-around time operator
#WA(t)
WA={}
for r in range(-1000,1):
    for t in D:
        arr = t+(T*r)
        WA[arr] = t

#%%
#########constraint##########


#(2)
#FI[i][j][t]
#웨하시점 재고
#현 시점 재고 = t-1시점 재고 + 웨하0에서 온거 - 리테일러 주문와서 보낸거 - 내가 다른 웨하에 보낸거 + 내가 다른 웨하에서 주문한거
for i in P:
    for j in W:
        for t in D:
            solver.Add(FI[i][j][t] == FI[i][j][WA[t-1]] + SQ[i]['Warehouse0'][j][WA[t-LTT['Warehouse0'][j]]] - 
                       sum(SQ[i][j][k][WA[t]] for k in R) - 
                       sum(SQ[i][j][l][WA[t]] for l in W if l !=j) +
                       sum(SQ[i][l][j][WA[t-LTT[l][j]]] for l in W if l !=j))
#(3)
#FI[i][k][t]
#리테일러 시점 재고
#현시점 재고 = t-1시점 재고 + 웨하에서 온거 - (고객수요 + 고객수요는 있지만 못보내준거) - 다른 리테일러한테 보낸거 + 다른 리테일러한테 주문한거 
for i in P:
    for k in R:
        for t in D:
            solver.Add(FI[i][k][t] == FI[i][k][WA[t-1]] + sum(SQ[i][j][k][WA[t-LTT[j][k]]] for j in W) - 
                       (CD[i][k][WA[t]] - LS[i][k][WA[t]]) -
                       sum(SQ[i][k][m][WA[t]] for m in R if m != k) +
                       sum(SQ[i][m][k][WA[t-LTT[m][k]]] for m in R if m != k))
     
#수요 만족 제약(임의로 넣은거임)#
#웨하 수요만족 제약
for i in P:
    for j in W:
        solver.Add(sum(SQ[i]['Warehouse0'][j][WA[t-LTT['Warehouse0'][j]]] for t in D) +
                   sum(SQ[i][l][j][WA[t-LTT[l][j]]] for l in W if l !=j for t in D) >=
                   sum(SQ[i][j][k][WA[t]] for k in R for t in D) +
                   sum(SQ[i][j][l][WA[t]] for l in W if l != j for t in D))


#리테일러 수요만족 제약
for i in P:
    for k in R:
        solver.Add(sum(SQ[i][j][k][WA[t-LTT[j][k]]] for j in W for t in D) -
                   sum(SQ[i][k][m][WA[t]] for m in R if m != k) +
                   sum(SQ[i][m][k][WA[t-LTT[m][k]]] for m in R if m != k for t in D) +
                   sum(LS[i][k][t] for t in D) >= 
                   sum(CD[i][k][WA[t]]-LS[i][k][WA[t]] for t in D))


#(4)
#웨하에서 주문이 일어났으면 BV1 존재
for i in P:
    for j in W:
        for t in D:
            solver.Add(SQ[i]['Warehouse0'][j][t] + sum(SQ[i][l][j][t] for l in W if l != j) <= 10000000 * BV1[i][j][t])

#(5)
#리테일러에서 주문이 일어났으면 BV1존재
for i in P:
    for k in R:
        for t in D:
            solver.Add(sum(SQ[i][j][k][t] for j in W) + sum(SQ[i][m][k][t] for m in R if m != k) <= 10000000 * BV1[i][k][t])



for i in P:
    for j in I:
        for k in I:
            if j !=k:
                for t in D:
                    solver.Add(SQ[i][j][k][t] <= 1000000 * BV2[i][j][k][t])




#(6)
#주문수량/배치사이즈 = 배치 수
for i in P:
    for j in I:
        for k in I:
            if j !=k:
                for t in D:
                    solver.Add(NumBat[i][j][k][t] * BatSiz == SQ[i][j][k][t])

#(7)
#재고는 한계용량을 넘을수 없다
for j in DN:
    for t in D:
        solver.Add(sum(FI[i][j][t] for i in P) <= STC[j][t])


#(8)
#한번에 보낼 수 있는 배송 용량을 초과할수 없다
for j in DN:
    for k in DN:
        if j!=k:
            for t in D:
                solver.Add(sum(SQ[i][j][k][t] for i in P) <= TRACMAX[j][k])

#(9)
# 최소 배송량보다는 많이 보내야 한다
for j in DN:
    for k in DN:
        if j!=k:
            for t in D:
                solver.Add(TRACMIN[j][k]*BV2[i][j][k][t] <= sum(SQ[i][j][k][t] for i in P))




for i in P:
    for j in I:
        for k in I:
            for t in D:
                if j in W0 and k in R:
                    solver.Add(SQ[i][j][k][t] == 0)
                elif j in W and k in W0:
                    solver.Add(SQ[i][j][k][t] == 0)
                elif j in R and k in W0:
                    solver.Add(SQ[i][j][k][t] == 0)
                elif j in R and k in W:
                    solver.Add(SQ[i][j][k][t] == 0)



#(10)
#재고는 안전재고 보다 많아야 한다
for i in P:
    for j in DN:
        for t in D:
            solver.Add(SS[i][j] <= FI[i][j][t])




W0_cost = sum((HTC[i]['Warehouse0'][k] * LTT['Warehouse0'][k] + TRC[i]['Warehouse0'][k]) * SQ[i]['Warehouse0'][k][t] for i in P for k in W for t in D)

W_cost = (sum(OC[i][j] * BV1[i][j][t] + HOC[i][j] * FI[i][j][t] for i in P for j in W for t in D) +
          sum((HTC[i][j][k] * LTT[j][k] + TRC[i][j][k]) *SQ[i][j][k][t] for i in P for j in W for k in DN for t in D))

R_cost = (sum(OC[i][j] * BV1[i][j][t] + HOC[i][j] * FI[i][j][t] + LSC[i][j][t] * LS[i][j][t] for i in P for j in R for t in D) +
          sum((HTC[i][j][k] *LTT[j][k] + TRC[i][j][k]) * SQ[i][j][k][t] for i in P for j in R for k in R for t in D))

solver.Minimize((W0_cost + W_cost + R_cost)*H/T)

#%%
W#%%
# 문제 풀기
solver.SetTimeLimit(time*1000)
status = solver.Solve()
# 현재 시간 가져오기
current_time = datetime.now()

# 포맷에 맞춰 출력
print("현재 시간:", current_time.strftime("%H:%M:%S"))
if status == pywraplp.Solver.OPTIMAL:
    print("최적해 입니다.")
else:
    print("최적해를 구하지 못했습니다.")
print('Solution:')
print(f'Objective value = {solver.Objective().Value()}')


#%%

# for i in P:
#     for j in I:
#         for t in D:
#             if BV1[i][j][t].solution_value()>0.1:
#                 print(f'BV1_{i}_{j}_{t} = {BV1[i][j][t].solution_value()}')

for i in P:
    for j in I:
        for t in D:
            if FI[i][j][t].solution_value()>0.1:
                print(f'FI_{i}_{j}_{t} = {FI[i][j][t].solution_value()}')

for i in P:
    for j in I:
        for t in D:
            if LS[i][j][t].solution_value()>0.1:
                print(f'LS_{i}_{j}_{t} = {LS[i][j][t].solution_value()}')

#%%
for i in P:
    for j in SN:
        for k in DN:
            for t in D:
                if SQ[i][j][k][t].solution_value()>0.1:
                    print(f'SQ_{i}_{j}_{k}_{t} = {SQ[i][j][k][t].solution_value()}')
#%%

# for i in P:
#     for j in I:
#         for k in I:
#             for t in D:
#                 if NumBat[i][j][k][t].solution_value()>0.1:
#                     print(f'NumBat_{i}_{j}_{k}_{t} = {NumBat[i][j][k][t].solution_value()}')


#%%
########재고######


#웨하 재고
Inv_W = {}
for t in D:
    Inv_W[t] = (sum(round(FI[i][j][t].solution_value()) for i in P for j in W))
#웨하 재고량 합 
SUM_Inv_W = (sum([Inv_W[t] for t in D]))
#웨하 재고량 평균
print('웨하재고량 평균')
InvAve_W = (SUM_Inv_W/T)
print(InvAve_W)
values_list = list(Inv_W.values())
values_array = np.array(values_list)
# 웨하 재고 분산
Var_Inv_W = np.var(values_array)



#리테 재고
Inv_R = {}
for t in D:
    Inv_R[t] = (sum(round(FI[i][j][t].solution_value()) for i in P for j in R))
#리테 재고량 합 
SUM_Inv_R = (sum([Inv_R[t] for t in D]))
#리테 재고량 평균
print('리테재고량 평균')
InvAve_R = (SUM_Inv_R/T)
print(InvAve_R)

values_list = list(Inv_R.values())
values_array = np.array(values_list)
# 리테 재고 분산
Var_Inv_R = np.var(values_array)


        
# 각 재고 데이터 딕셔너리를 데이터프레임으로 변환
inventory_data = {
    'Warehouse': Inv_W,
    'Retail': Inv_R
}

#%%
# 딕셔너리 키를 데이터프레임으로 변환하여 각 행을 출처로 지정
df_inventory = pd.DataFrame.from_dict(inventory_data, orient='index')

# df_inventory 데이터프레임을 사용하여 시각화
df_inventory.plot(kind='bar', figsize=(10, 6), width=0.8)

# 그래프 제목과 축 설정
plt.title(f"Inventory", fontsize=16)
plt.xlabel('Inventory Source', fontsize=12)
plt.ylabel('Inventory Amount', fontsize=12)

# 가로축 레이블을 열 이름으로 설정
plt.xticks(ticks=range(len(df_inventory.index)), labels=df_inventory.index, rotation=0)
# 각각의 x축을 점선으로 구분
for x in range(len(df_inventory.index)):
    plt.axvline(x=x - 0.5, color='gray', linewidth=1)


# 범례를 각 기간으로 설정
plt.legend(title='Period', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0,100)

# 그래프 그리기
plt.tight_layout()
plt.show()

#%%

#웨하 주문
ORD_W = {}
for t in D:
    ORD_W[t] = (sum(round(SQ[i][j][k][t].solution_value()) for i in P for j in W0 for k in W) + sum(round(SQ[i][l][j][t].solution_value()) for i in P for l in W for j in W if l != j))
#웨하 주문합
SUM_ORD_W = (sum([ORD_W[t] for t in D]))
#웨하 주문 평균
ORDAve_W = (SUM_ORD_W/T)

values_list = list(ORD_W.values())
values_array = np.array(values_list)
# 웨하 주문 분산
Var_ORD_W = np.var(values_array)



#리테 주문
ORD_R = {}
for t in D:
    ORD_R[t] = (sum(round(SQ[i][j][k][t].solution_value()) for i in P for j in W for k in R) + sum(round(SQ[i][l][j][t].solution_value()) for i in P for l in R for j in R if l != j))
#리테 주문 합
SUM_ORD_R = (sum([ORD_R[t] for t in D]))
#리테 주문 평균
ORDAve_R = (SUM_ORD_R/T)
values_list = list(ORD_R.values())
values_array = np.array(values_list)
# 웨하 주문 분산
Var_ORD_R = np.var(values_array)



#고객 수요
CdeR = {}
for t in D:
    CdeR[t] = (sum(CD[i][k][t] for i in P for k in R))


#고객 주문합
SUM_CdeR = (sum([CdeR[t] for t in D]))
#고객 주문 평균
CdeAveR = (SUM_CdeR/T)

values_list = list(CdeR.values())
values_array = np.array(values_list)
#고객 주문 분산
CdeVarR = np.var(values_array)

# 각 데이터 출처에 대한 딕셔너리들을 데이터프레임으로 변환
orders_data = {
    'Warehouse': ORD_W,
    'Retailer': ORD_R,
    'Customer': CdeR
}
        


# 딕셔너리 키를 데이터프레임으로 변환하여 각 행을 출처로 지정
df_orders = pd.DataFrame.from_dict(orders_data, orient='index')

# 데이터프레임을 사용하여 시각화
df_orders.plot(kind='bar', figsize=(10, 6), width=0.8)

# 그래프 제목과 축 설정
plt.title(f"Order", fontsize=16)
plt.xlabel('Order Source', fontsize=12)
plt.ylabel('Order Amount', fontsize=12)

# 가로축 레이블을 열 이름으로 설정
plt.xticks(ticks=range(len(df_orders.index)), labels=df_orders.index, rotation=0)

# 각각의 x축을 점선으로 구분
for x in range(len(df_orders.index)):
    plt.axvline(x=x - 0.5, color='gray', linewidth=1)


# 범례를 각 기간으로 설정
plt.legend(title='Period', bbox_to_anchor=(1.05, 1), loc='upper left')

# 그래프 그리기
plt.tight_layout()
plt.show()

#%%

if InvAve_W!=0:
    InvVarRatW = (Var_Inv_W/InvAve_W)/(CdeVarR/CdeAveR)
else:
    InvVarRatW = 0
print('웨하 재고 분산 레이트')
print(InvVarRatW)

if InvAve_R!=0:
    InvVarRatR = (Var_Inv_R/InvAve_R)/(CdeVarR/CdeAveR)
else:
    InvVarRatR = 0
print('리테 재고 분산 레이트')
print(InvVarRatR)





OrdVarRatW = (Var_ORD_W/ORDAve_W)/(CdeVarR/CdeAveR)
print('웨하 오더 분산 레이트')
print(OrdVarRatW)


OrdVarRatR = (Var_ORD_R/ORDAve_R)/(CdeVarR/CdeAveR)
print('리테 오더 분산 레이트')
print(OrdVarRatR)




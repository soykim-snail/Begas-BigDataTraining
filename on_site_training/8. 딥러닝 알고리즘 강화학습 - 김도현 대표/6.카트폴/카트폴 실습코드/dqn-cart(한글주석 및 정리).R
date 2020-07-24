#======================================================================
# CartPole with DQN
#
# Reference : 
#       AG BARTO(1983)
#       OpenAI gym
#       파이썬과 케라스로 배우는 강화학습(이용우, 양혁렬 외 3명 지음)
#
# Author : Yongjun Jo
#
# BEGAS
#======================================================================


#======================================================================
# 카트폴 DQN 환경 설정
#======================================================================

#환경 초기화 및 garbage collection 
rm(list=ls());gc(reset=T)

#작업공간 설정
wd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wd)

#패키지 설치유무 확인 및 로드 
pack.list <- c('reticulate', 'keras', 'abind', 'data.table', 'ggplot2')
load <- sapply(pack.list,require,character.only = T)

#패키지가 없을 시 설치
#   패키지가 없어서 설치가 필요하면 proceed? [y/n] 메세지가 뜰 때 콘솔에 y를 입력하세요
if(FALSE %in% load){
  cat('Need to install packages :',pack.list[load==F],'\n')
  if(readline('proceed? [y/n] :') == 'y'){
    install.packages(pack.list[load==F])
    sapply(pack.list[load==F],require,character.only = T)
  }else{
    cat('-- Install denied --')
  }
}



#함수 스크립트에서 함수 로딩 
source("./dqn-cart-fn.R")

# 콘다 환경 리스트 확인
conda_list()

# 설정된 콘다 환경 불러오기 : rl_env 
use_condaenv('rl_env')

#파이썬에서 gym패키지 로드 및 해당 패키지의 게임 환경 확인 
obj_gym <- import('gym')
list.game <- head(names(obj_gym$envs$registry$env_specs)); print(list.game)

# 카트폴 환경 설정
env            <- obj_gym$make('CartPole-v1')

# q근사를 위한 모델과 타겟 근사를 위한 모델 생성 
q.aprx.crtp     <- func.q.aprx.gen()
q.trgt.crtp     <- func.q.aprx.gen()

#타겟 모델 업데이트
func.upd.trgt(q.trgt = q.trgt.crtp, q.aprx = q.aprx.crtp)

# 글로벌 파라미터 지정 

# Replay memory 생성(max = 2000)
dt.rply.mem <- data.table(S1   = as.numeric(), S2  = as.numeric(), S3  = as.numeric(), S4   = as.numeric(),
                          A    = as.integer(),
                          R    = as.numeric(),
                          SP1  = as.numeric(), SP2  = as.numeric(), SP3  = as.numeric(), SP4  = as.numeric(),
                          TRML = as.logical())[1:2000]

#에피소드별 현황 테이블 설정 : Status
dt.stat.ep  <- data.table(NUM_EP       = as.numeric(),
                          SCORE        = as.numeric(),
                          STEP_EP      = as.numeric(),
                          STEP_GLO     = as.numeric(),
                          AVG_MAX_Q    = as.numeric(),
                          AVG_LOSS     = as.numeric(),
                          LEARING_TIME = as.character())[1:120]

#카트폴 학습 하이퍼 파라미터 설정
glo.ctrl.para <- list(epsilon       = 1,
                      glo.step      = 0,
                      tr.strt.step  = 1000,
                      check.cnvrge  = NA,
                      cnvrge        = F,
                      glo.strt.time = Sys.time(),
                      glo.end.time  = Sys.time())

#랜덤시드 설정 
set.seed(1224)




#======================================================================
# 카트폴 DQN 학습 
#======================================================================


#에피소드 학습 
for(e in 1:120){
  ep.ctrl.para <-list(score=0, trml=F, ep.step=0,
                      qmax =0, loss=0, 
                      strt.time=Sys.time(),end.time=Sys.time())
  
  #에피소드마다 환경 리셋
  s <- env$reset()
  
  while(!ep.ctrl.para$trml){ #터미널(trml)이 TRUE(=게임이 끝남) 이 될때까지 루프
    
    #step을 몇번 진행했는지 체크
    glo.ctrl.para$glo.step <- glo.ctrl.para$glo.step + 1
    ep.ctrl.para$ep.step   <- ep.ctrl.para$ep.step   + 1
    
    # 입실론-그리디 정책에 따라 행동 선택 
    if(runif(n = 1, min = 0, max = 1) <= glo.ctrl.para$epsilon){
      a <- as.integer(sample(size = 1, x = c(0, 1))) #랜덤 행동 
    }else{
      tmp.a  <- which.max(q.aprx.crtp$predict(t(s))) #q 값 중 최대값 선택
      a      <- as.integer(tmp.a-1)
    }
    
    # 액션에 따른 time-step 진행 및 진행결과 업데이트(SARS')
    observe            <- env$step(a)
    sp                 <- observe[[1]] 
    ep.ctrl.para$trml  <- observe[[3]] # terminal
    tmp.r              <- observe[[2]] # reward
    r                  <- ifelse(!ep.ctrl.para$trml|ep.ctrl.para$score==499, tmp.r, -100)
    ep.ctrl.para$score <- ep.ctrl.para$score + r
    
    
    # SARS'와 게임 결과를 메모리에 저장
    if(glo.ctrl.para$glo.step<=2000){ #메모리가 2천개 미만일시 업데이트
      dt.rply.mem[glo.ctrl.para$glo.step,
                  c('S1','S2','S3','S4','A','R','SP1','SP2','SP3','SP4','TRML'):=
                    list(s[1],s[2],s[3],s[4],
                         a, r,
                         sp[1],sp[2],sp[3],sp[4],
                         ep.ctrl.para$trml)]
    }else{
      dt.rply.mem <- dt.rply.mem[c(2:2000,1),] #메모리가 2천개 이상일시 앞의 메모리를 지우고 업데이트
      dt.rply.mem[2000,
                  c('S1','S2','S3','S4','A','R','SP1','SP2','SP3','SP4','TRML'):=
                    list(s[1],s[2],s[3],s[4],
                         a, r,
                         sp[1],sp[2],sp[3],sp[4],
                         ep.ctrl.para$trml)]
    }
    
    
    # 메모리가 1000개 이상 쌓이면 학습 실시 
    if(glo.ctrl.para$glo.step >= glo.ctrl.para$tr.strt.step){
      # 입실론을 epsilon.decay 만큼 감소
      glo.ctrl.para$epsilon  <- glo.ctrl.para$epsilon %>% func.decay.e()
      # 모델 학습 및 loss 저장 
      tmp.loss <-func.tr.mdl(discount.fctr = 0.99,
                             rply.mem      = dt.rply.mem,
                             glo.step      = ifelse(glo.ctrl.para$glo.step<=2000, 
                                                    glo.ctrl.para$glo.step, 2000),
                             size.batch    = 64L,
                             q.trgt        = q.trgt.crtp,
                             q.aprx        = q.aprx.crtp,
                             size.state    = 4L,
                             size.action   = 3L)
      # 학습 결과를 업데이트
      ep.ctrl.para$loss <- ep.ctrl.para$loss + tmp.loss
      ep.ctrl.para$qmax <- ep.ctrl.para$qmax + max(q.aprx.crtp$predict(t(s)))
    }
    
    # S'를 S에 업데이트
    s <- sp
    
    # 에피소드마다 정보 업데이트
    if(ep.ctrl.para$trml){
      ep.ctrl.para$end.time <- Sys.time()
      # 타겟 Q함수의 가중치를 Q근사 모델의 가중치로 업데이트함
      func.upd.trgt(q.trgt = q.trgt.crtp, q.aprx = q.aprx.crtp)
      
      #최종 스코어 및 Max Q, loss 등의 정보를 업데이트 
      score        <- ifelse(ep.ctrl.para$score==500, ep.ctrl.para$score, ep.ctrl.para$score+100)
      avg.q.max    <- ep.ctrl.para$qmax / ep.ctrl.para$ep.step
      avg.loss     <- ep.ctrl.para$loss / ep.ctrl.para$ep.step
      tmp.timediff <- ep.ctrl.para$end.time - ep.ctrl.para$strt.time
      learningtime <- paste0(round(tmp.timediff, 2), ' ', attributes(tmp.timediff)$units)
      
      #해당 에피소드 정보 콘솔에 출력
      func.dash.board(ep       = e,
                      score    = score,
                      q.max    = avg.q.max,
                      loss     = avg.loss,
                      lrn.time = learningtime)
      
      #에피소드별 정보 수집 테이블 업데이트
      dt.stat.ep[e, 
                 c('NUM_EP','SCORE','STEP_EP','STEP_GLO','AVG_MAX_Q','AVG_LOSS','LEARING_TIME'):=
                   list(e,score,ep.ctrl.para$ep.step,
                        glo.ctrl.para$glo.step,avg.q.max,avg.loss,learningtime)]
      
      #스코어가 수렴하는지 확인 
      glo.ctrl.para$check.cnvrge[ifelse(mod(e,5)!=0,mod(e,5),5)] <- score 
      if(length(glo.ctrl.para$check.cnvrge)==5){
        if(mean(glo.ctrl.para$check.cnvrge) >= 480){
          cat('Converge!\n');
          glo.ctrl.para$cnvrge <- T
          break;
        }
      }
    }
    
    #매 20번째 에피소드마다 모델 저장 
    if(mod(e, 20)==0){
      if(!dir.exists('./crtp/model')){dir.create('./crtp/model', showWarnings = F, recursive = T)}
      if(!dir.exists('./crtp/glopara')){dir.create('./crtp/glopara', showWarnings = F, recursive = T)}
      if(!dir.exists('./crtp/stat')){dir.create('./crtp/stat', showWarnings = F, recursive = T)}
      if(!dir.exists('./crtp/rplymem')){dir.create('./crtp/rplymem', showWarnings = F, recursive = T)}
      saveRDS(q.aprx.crtp$get_weights(), sprintf('./crtp/model/qaprx_mdl_%d.rds', e))
      saveRDS(glo.ctrl.para, sprintf('./crtp/glopara/global_para_%d.rds', e))
      fwrite(na.omit(dt.rply.mem), sprintf('./crtp/rplymem/replay_mem_%d.csv', e))
      fwrite(na.omit(dt.stat.ep), sprintf('./crtp/stat/stat_episode_%d.csv', e))
      png(sprintf('./crtp/stat/stat_%d.png', e),width=1000,height=600)
      par(mfrow=c(2,1))
      tmp.q <- na.omit(dt.stat.ep[,AVG_MAX_Q])
      tmp.s <- na.omit(dt.stat.ep[,SCORE])
      plot(tmp.q, main='AVERAGE OF MAX Q',ylab='max Q', xlab='eqisode', type='l')
      plot(tmp.s, main='SCORE BY EPISODE',ylab='score', xlab='eqisode', type='l')
      dev.off()
    }
    
    #수렴 시 학습 종료
    if(glo.ctrl.para$cnvrge){break}
  }
}

#학습 종료 시간 저장 
glo.ctrl.para$glo.end.time <- Sys.time()



#======================================================================
# 학습결과 작업공간에 저장 및 시각화  
#======================================================================

#가중치와 파라미터 저장 
saveRDS(q.aprx.crtp$get_weights(), sprintf('./crtp/model/qaprx_mdl_fin.rds'))
saveRDS(glo.ctrl.para, sprintf('./crtp/glopara/global_para_fin.rds'))

#리플레이 메모리와 에피소드 결과 저장 
fwrite(dt.rply.mem, sprintf('./crtp/rplymem/replay_mem_fin.csv'))
fwrite(dt.stat.ep, sprintf('./crtp/stat/stat_episode_fin.csv'))

#마지막 학습 결과 시각화 
png(sprintf('./crtp/stat/stat_fin.png', e),width=800,height=500)

#에피소드별 학습 진행과정 시각화
tmp.q <- na.omit(dt.stat.ep[,AVG_MAX_Q])
tmp.s <- na.omit(dt.stat.ep[,SCORE])
plt <- data.table(value   = c(tmp.q, tmp.s),
                  class   = c(rep('AVERAGE OF MAX Q', length(tmp.q)), 
                              rep('SCORE BY EPISODE', length(tmp.s))),
                  episode = c(seq_len(length(tmp.q)), seq_len(length(tmp.s)))) %>%
  ggplot(aes(x = episode, y = value)) +
  geom_line(alpha=0.3, col='blue', size=1) +
  facet_wrap(~class, scale='free') +
  theme_bw() +
  labs(title='RESULT BY EPISODE')
print(plt)

#종료
dev.off()

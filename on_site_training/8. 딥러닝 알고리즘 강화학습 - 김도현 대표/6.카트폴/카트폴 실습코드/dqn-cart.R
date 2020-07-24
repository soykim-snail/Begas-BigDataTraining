rm(list=ls());gc(reset=T)
wd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wd)
############
# packages #
############
pack.list <- c('reticulate', 'keras', 'abind', 'data.table', 'ggplot2')
for(x in pack.list){
  if(!x%in%installed.packages()){
    install.packages(x)
  }else{
    cat(sprintf('%s already exists.\n',x))
  }
}



library(reticulate);library(keras);
library(abind);library(data.table);library(ggplot2)
source("./dqn-cart-fn.R")

# Check conda environment.
conda_list()
# Activate virtual environment.
use_condaenv('rl_env')
obj_gym <- import('gym')
list.game <- head(names(obj_gym$envs$registry$env_specs)); print(list.game)
# Environment
env            <- obj_gym$make('CartPole-v1')

# Define q
q.aprx.crtp     <- func.q.aprx.gen()
q.trgt.crtp     <- func.q.aprx.gen()
func.upd.trgt(q.trgt = q.trgt.crtp, q.aprx = q.aprx.crtp)

# Set up global parameters.
# Replay memory
dt.rply.mem <- data.table(S1   = as.numeric(), S2  = as.numeric(), S3  = as.numeric(), S4   = as.numeric(),
                          A    = as.integer(),
                          R    = as.numeric(),
                          SP1  = as.numeric(), SP2  = as.numeric(), SP3  = as.numeric(), SP4  = as.numeric(),
                          TRML = as.logical())[1:2000]

# Status
dt.stat.ep  <- data.table(NUM_EP       = as.numeric(),
                          SCORE        = as.numeric(),
                          STEP_EP      = as.numeric(),
                          STEP_GLO     = as.numeric(),
                          AVG_MAX_Q    = as.numeric(),
                          AVG_LOSS     = as.numeric(),
                          LEARING_TIME = as.character())[1:120]

glo.ctrl.para <- list(epsilon       = 1,
                      glo.step      = 0,
                      tr.strt.step  = 1000,
                      check.cnvrge  = NA,
                      cnvrge        = F,
                      glo.strt.time = Sys.time(),
                      glo.end.time  = Sys.time())
set.seed(1224)

for(e in 1:120){
  ep.ctrl.para <-list(score=0, trml=F, ep.step=0,
                      qmax =0, loss=0, 
                      strt.time=Sys.time(),end.time=Sys.time())
  
  # Reset environment
  # We start the new game every episode.
  s <- env$reset()
  
  while(!ep.ctrl.para$trml){
    glo.ctrl.para$glo.step <- glo.ctrl.para$glo.step + 1
    ep.ctrl.para$ep.step   <- ep.ctrl.para$ep.step   + 1
    
    # Select action
    if(runif(n = 1, min = 0, max = 1) <= glo.ctrl.para$epsilon){
      a <- as.integer(sample(size = 1, x = c(0, 1)))
    }else{
      tmp.a  <- which.max(q.aprx.crtp$predict(t(s)))
      a      <- as.integer(tmp.a-1)
    }
    
    # Take action
    observe            <- env$step(a)
    sp                 <- observe[[1]] 
    ep.ctrl.para$trml  <- observe[[3]] # terminal
    tmp.r              <- observe[[2]] # reward
    r                  <- ifelse(!ep.ctrl.para$trml|ep.ctrl.para$score==499, tmp.r, -100)
    ep.ctrl.para$score <- ep.ctrl.para$score + r
    
    
    # Assign sar's' & terminal to memory.
    if(glo.ctrl.para$glo.step<=2000){
      dt.rply.mem[glo.ctrl.para$glo.step,
                  c('S1','S2','S3','S4','A','R','SP1','SP2','SP3','SP4','TRML'):=
                    list(s[1],s[2],s[3],s[4],
                         a, r,
                         sp[1],sp[2],sp[3],sp[4],
                         ep.ctrl.para$trml)]
    }else{
      dt.rply.mem <- dt.rply.mem[c(2:2000,1),]
      dt.rply.mem[2000,
                  c('S1','S2','S3','S4','A','R','SP1','SP2','SP3','SP4','TRML'):=
                    list(s[1],s[2],s[3],s[4],
                         a, r,
                         sp[1],sp[2],sp[3],sp[4],
                         ep.ctrl.para$trml)]
    }
    
    
    # Train model after getting memory more than 1000.
    if(glo.ctrl.para$glo.step >= glo.ctrl.para$tr.strt.step){
      # decay exploration rate
      glo.ctrl.para$epsilon  <- glo.ctrl.para$epsilon %>% func.decay.e()
      # Train
      tmp.loss <-func.tr.mdl(discount.fctr = 0.99,
                             rply.mem      = dt.rply.mem,
                             glo.step      = ifelse(glo.ctrl.para$glo.step<=2000, 
                                                    glo.ctrl.para$glo.step, 2000),
                             size.batch    = 64L,
                             q.trgt        = q.trgt.crtp,
                             q.aprx        = q.aprx.crtp,
                             size.state    = 4L,
                             size.action   = 3L)
      # Write info
      ep.ctrl.para$loss <- ep.ctrl.para$loss + tmp.loss
      ep.ctrl.para$qmax <- ep.ctrl.para$qmax + max(q.aprx.crtp$predict(t(s)))
    }
    
    # The current state take over to next states.
    s <- sp
    
    # Write information of learning by each episode.
    if(ep.ctrl.para$trml){
      ep.ctrl.para$end.time <- Sys.time()
      # Weights of trgt q function is updated via weights of aprx q.
      func.upd.trgt(q.trgt = q.trgt.crtp, q.aprx = q.aprx.crtp)
      
      score        <- ifelse(ep.ctrl.para$score==500, ep.ctrl.para$score, ep.ctrl.para$score+100)
      avg.q.max    <- ep.ctrl.para$qmax / ep.ctrl.para$ep.step
      avg.loss     <- ep.ctrl.para$loss / ep.ctrl.para$ep.step
      tmp.timediff <- ep.ctrl.para$end.time - ep.ctrl.para$strt.time
      learningtime <- paste0(round(tmp.timediff, 2), ' ', attributes(tmp.timediff)$units)
      
      func.dash.board(ep       = e,
                      score    = score,
                      q.max    = avg.q.max,
                      loss     = avg.loss,
                      lrn.time = learningtime)
      # Write info.
      dt.stat.ep[e, 
                 c('NUM_EP','SCORE','STEP_EP','STEP_GLO','AVG_MAX_Q','AVG_LOSS','LEARING_TIME'):=
                   list(e,score,ep.ctrl.para$ep.step,
                        glo.ctrl.para$glo.step,avg.q.max,avg.loss,learningtime)]
      
      # Check convergence.
      glo.ctrl.para$check.cnvrge[ifelse(mod(e,5)!=0,mod(e,5),5)] <- score 
      if(length(glo.ctrl.para$check.cnvrge)==5){
        if(mean(glo.ctrl.para$check.cnvrge) >= 480){
          cat('Converge!\n');
          glo.ctrl.para$cnvrge <- T
          break;
        }
      }
    }
    
    # Save model every 20 episode.
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
    if(glo.ctrl.para$cnvrge){break}
  }
}

glo.ctrl.para$glo.end.time <- Sys.time()
saveRDS(q.aprx.crtp$get_weights(), sprintf('./crtp/model/qaprx_mdl_fin.rds'))
saveRDS(glo.ctrl.para, sprintf('./crtp/glopara/global_para_fin.rds'))
fwrite(na.omit(dt.rply.mem), sprintf('./crtp/rplymem/replay_mem_fin.csv'))
fwrite(na.omit(dt.stat.ep), sprintf('./crtp/stat/stat_episode_fin.csv'))
#png(sprintf('./crtp/stat/stat_fin.png', e),width=800,height=500)
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
dev.off()

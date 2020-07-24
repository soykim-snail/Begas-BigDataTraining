library(reticulate);library(imager);library(keras);
library(abind);library(data.table);
#################
# DQN functions #
#################
# q(s,a,w)
func.q.aprx.gen <- function(size.state  = 4L,
                            size.action = 2L){
  frm.mdl  <- layer_input(shape=size.state)
  dnse1    <- layer_dense(frm.mdl, units = 24, activation = 'relu', kernel_initializer = initializer_he_uniform(seed=1223))
  dnse2    <- layer_dense(dnse1, units = 24, activation = 'relu', kernel_initializer = initializer_he_uniform(seed=1224))
  output   <- layer_dense(dnse2, units = size.action, kernel_initializer = initializer_he_uniform(seed=1225))
  
  model    <- keras_model(inputs = frm.mdl, outputs= output)

  model$compile(loss      = 'mse',
                optimizer = optimizer_adam(lr = 0.001))

  return(model)
}

# Decay exploration parameter (epsilon)
func.decay.e <- function(epsilon){
  # Maintenance
  # epsilon <- 1
  if(epsilon>0.01){
    return(epsilon*0.999)
  }else{
    return(0.01)
  }
}

# Training
func.tr.mdl <- function(discount.fctr = 0.99,
                        rply.mem      = NULL,
                        glo.step      = NULL,
                        size.batch    = 64L,
                        q.trgt        = NULL,
                        q.aprx        = NULL,
                        size.state    = 4L,
                        size.action   = 3L){
  # Maintenance
  # for(i in 1:100){ dt.rply.mem[i,
  #                              c('S1','S2','S3','S4','A','R','SP1','SP2','SP3','SP4','TRML'):=
  #                                list(s[1],s[2],s[3],s[4],
  #                                     a, r,
  #                                     sp[1],sp[2],sp[3],sp[4],
  #                                     trml)]}
  # glo.step<-100
  nbr.sample    <- sample.int(n=glo.step, size=size.batch)
  dt.batch      <- rply.mem[nbr.sample,]
  batch.s       <- dt.batch[,list(S1,S2,S3,S4)] %>% as.matrix()
  batch.a       <- dt.batch[, A]
  batch.r       <- dt.batch[, R]
  batch.sp      <- dt.batch[,list(SP1,SP2,SP3,SP4)] %>% as.matrix()
  batch.trml    <- dt.batch[ , TRML]
  
  # Make labels from target q func.
  tmp.q.trgt.val     <- q.aprx$predict(batch.s)
  tmp.q.trgt.val2    <- q.trgt$predict(batch.sp)
  batch.q.trgt.value <- tmp.q.trgt.val
  for(nbr in seq_len(size.batch)){
    if(batch.trml[nbr]){
      batch.q.trgt.value[nbr, batch.a[nbr]+1] <- batch.r[nbr]
    }else{
      batch.q.trgt.value[nbr, batch.a[nbr]+1] <- batch.r[nbr] + discount.fctr*max(tmp.q.trgt.val2[nbr,])
    }
  }
  
  # training
  q.aprx$fit(batch.s, batch.q.trgt.value, epochs = 1L,
             batch_size = size.batch, verbose = 0)
  loss     <- q.aprx$evaluate(batch.s, batch.q.trgt.value, verbose=0)
  
  rm(batch.a, batch.q.trgt.value, batch.r, batch.s, batch.sp, batch.trml, dt.batch)
  return(loss)
}

# Update target.
func.upd.trgt <- function(q.trgt = NULL,
                          q.aprx = NULL){
  cat('Weights of approximation q function are assigned to target q function.\n')
  q.trgt$set_weights(q.aprx$get_weights())
}

# Dash Board
func.dash.board <- function(ep       = NULL,
                            score    = NULL,
                            q.max    = NULL,
                            loss     = NULL,
                            lrn.time = NULL){
  cat('================================\n')
  cat('===========DASH BOARD===========\n')
  cat('================================\n')
  cat(sprintf('EPISODE           : %d\n', e))
  cat(sprintf('SCORE             : %d\n', score))
  cat(sprintf('AVERAGE OF Q MAX  : %f\n', q.max))
  cat(sprintf('AVERAGE OF LOSS   : %f\n', loss))
  cat(sprintf('LEARNING TIME     : %s\n', lrn.time))
}

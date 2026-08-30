/**
 * Executable design model for Agent Canvas model-selection authority.
 *
 * This specification models the contract shared by the Home model selector,
 * conversation creation/reopen, and live model switching. It does not execute
 * the React implementation. Keep the transitions and the workflow path filters
 * in sync when those product semantics change.
 *
 * Authority rules:
 *   - Home displays the current account default.
 *   - An open conversation displays its persisted, running model.
 *   - Changing the account default does not change an open conversation.
 *   - A live switch updates the persisted, running, and displayed model together.
 *
 * The intentionally stale reopen and switch actions are negative controls. They
 * must violate displayMatchesAuthority, proving that the invariant can detect
 * the regressions it is intended to guard against.
 */
module agent_canvas_model_selection {
  type Model = ModelA | ModelB
  type Surface = Home | Conversation

  var accountDefault: Model
  var persistedConversationModel: Model
  var runningConversationModel: Model
  var displayedModel: Model
  var surface: Surface

  action init = all {
    accountDefault' = ModelA,
    persistedConversationModel' = ModelA,
    runningConversationModel' = ModelA,
    displayedModel' = ModelA,
    surface' = Home,
  }

  action setAccountDefault(model: Model): bool = all {
    accountDefault' = model,
    persistedConversationModel' = persistedConversationModel,
    runningConversationModel' = runningConversationModel,
    displayedModel' = if (surface == Home) model else displayedModel,
    surface' = surface,
  }

  action createConversation = all {
    surface == Home,
    accountDefault' = accountDefault,
    persistedConversationModel' = accountDefault,
    runningConversationModel' = accountDefault,
    displayedModel' = accountDefault,
    surface' = Conversation,
  }

  action returnHome = all {
    surface == Conversation,
    accountDefault' = accountDefault,
    persistedConversationModel' = persistedConversationModel,
    runningConversationModel' = runningConversationModel,
    displayedModel' = accountDefault,
    surface' = Home,
  }

  action reopenConversation = all {
    surface == Home,
    accountDefault' = accountDefault,
    persistedConversationModel' = persistedConversationModel,
    runningConversationModel' = persistedConversationModel,
    displayedModel' = persistedConversationModel,
    surface' = Conversation,
  }

  action switchConversationModel(model: Model): bool = all {
    surface == Conversation,
    accountDefault' = accountDefault,
    persistedConversationModel' = model,
    runningConversationModel' = model,
    displayedModel' = model,
    surface' = Conversation,
  }

  action step = any {
    setAccountDefault(ModelA),
    setAccountDefault(ModelB),
    createConversation,
    returnHome,
    reopenConversation,
    switchConversationModel(ModelA),
    switchConversationModel(ModelB),
  }

  val displayMatchesAuthority = {
    if (surface == Home) {
      displayedModel == accountDefault
    } else {
      and {
        runningConversationModel == persistedConversationModel,
        displayedModel == runningConversationModel,
      }
    }
  }

  // This Home state is reachable after creating a ModelA conversation,
  // switching it to ModelB, and returning Home.
  action initStaleReopenWitness = all {
    accountDefault' = ModelA,
    persistedConversationModel' = ModelB,
    runningConversationModel' = ModelB,
    displayedModel' = ModelA,
    surface' = Home,
  }

  action reopenWithStaleDisplay = all {
    surface == Home,
    accountDefault' = accountDefault,
    persistedConversationModel' = persistedConversationModel,
    runningConversationModel' = persistedConversationModel,
    displayedModel' = displayedModel,
    surface' = Conversation,
  }

  action initStaleSwitchWitness = all {
    accountDefault' = ModelA,
    persistedConversationModel' = ModelA,
    runningConversationModel' = ModelA,
    displayedModel' = ModelA,
    surface' = Conversation,
  }

  action switchWithStaleDisplay = all {
    surface == Conversation,
    accountDefault' = accountDefault,
    persistedConversationModel' = ModelB,
    runningConversationModel' = ModelB,
    displayedModel' = displayedModel,
    surface' = Conversation,
  }

  run homeDisplaysAccountDefaultTest =
    init
      .then(setAccountDefault(ModelB))
      .expect(and {
        displayedModel == ModelB,
        displayMatchesAuthority,
      })

  run conversationStartsWithAccountDefaultTest =
    init
      .then(setAccountDefault(ModelB))
      .then(createConversation)
      .expect(and {
        persistedConversationModel == ModelB,
        runningConversationModel == ModelB,
        displayedModel == ModelB,
        displayMatchesAuthority,
      })

  run defaultChangeDoesNotChangeOpenConversationTest =
    init
      .then(createConversation)
      .then(setAccountDefault(ModelB))
      .expect(and {
        accountDefault == ModelB,
        persistedConversationModel == ModelA,
        displayedModel == ModelA,
        displayMatchesAuthority,
      })

  run reopenedConversationDisplaysPersistedModelTest =
    init
      .then(createConversation)
      .then(switchConversationModel(ModelB))
      .then(returnHome)
      .then(reopenConversation)
      .expect(and {
        persistedConversationModel == ModelB,
        runningConversationModel == ModelB,
        displayedModel == ModelB,
        displayMatchesAuthority,
      })

  run liveSwitchUpdatesAllConversationModelsTest =
    init
      .then(createConversation)
      .then(switchConversationModel(ModelB))
      .expect(and {
        persistedConversationModel == ModelB,
        runningConversationModel == ModelB,
        displayedModel == ModelB,
        displayMatchesAuthority,
      })

  run staleReopenViolatesInvariantTest =
    init
      .then(createConversation)
      .then(switchConversationModel(ModelB))
      .then(returnHome)
      .expect(and {
        persistedConversationModel == ModelB,
        displayedModel == ModelA,
        displayMatchesAuthority,
      })
      .then(reopenWithStaleDisplay)
      .expect(not(displayMatchesAuthority))

  run staleSwitchViolatesInvariantTest =
    init
      .then(createConversation)
      .then(switchWithStaleDisplay)
      .expect(not(displayMatchesAuthority))
}
